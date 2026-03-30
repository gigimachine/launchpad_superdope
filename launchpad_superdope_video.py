import os
import glob
import time
import random
import math
import re
import yt_dlp
import librosa
import pygame
import mido
import pandas as pd
import numpy as np
import soundfile as sf

# ▼ 追加：動画再生用のOpenCV（インストールされていない場合は無視して進む安全設計）
try:
    import cv2
except ImportError:
    cv2 = None
    print("[!] cv2 (opencv-python) がインストールされていません。背景動画機能はオフになります。")

# ==========================================
# 0. 基本設定 & サビ・カッター設定
# ==========================================
try:
    available_ports = mido.get_input_names()
    PORT_NAME = next((p for p in available_ports if 'Launchpad' in p), None)
except Exception:
    PORT_NAME = None

if PORT_NAME is None:
    print("\n[エラー] Launchpadが見つかりません。USB接続を確認してください。")
    print(f"認識されているポート: {mido.get_input_names()}")
    exit()
else:
    print(f"\n[+] 使用ポート: {PORT_NAME}")

SONGS_DIR = 'songs'
FALL_TIME = 1.5

NUM_CLIPS = 6
CLIP_BEATS = 4 
EVAL_BEATS = 4 
STEP_BEATS = 0.25 
FREQ_MIN = 1000 
FREQ_MAX = 10000 
VOCAL_MARGIN = 1.0 
IGNORE_START_BEATS = 16  
IGNORE_END_BEATS = 32    
TRIM_DB = 30 

# 背景動画の透明度（0〜255）
VIDEO_ALPHA = 50 

# ==========================================
# 1. 楽曲ダウンロード＆フル解析エンジン
# ==========================================
def download_and_analyze():
    url = input("\n[+] YouTubeのURLを入力してください: ").strip()
    if not url:
        return

    os.makedirs(SONGS_DIR, exist_ok=True)
    
    print("\n【1】 楽曲情報を取得しています...")
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            raw_title = info.get('title', 'Unknown_Song')
        except Exception as e:
            print(f"[エラー] 楽曲情報の取得に失敗しました: {e}")
            return

    safe_title = re.sub(r'[\\/*?:"<>|]', "", raw_title)
    safe_title = re.sub(r'[\r\n\t]', "", safe_title).strip()
    if not safe_title:
        safe_title = "Unknown_Song"
        
    song_dir = os.path.join(SONGS_DIR, safe_title)
    audio_file = os.path.join(song_dir, 'audio.wav')
    csv_file = os.path.join(song_dir, 'pattern.csv')

    print(f" -> 処理対象: {safe_title}")
    print("【2】 動画と音声をダウンロードしています...")
    
    # ▼▼▼ 修正：AV1を絶対に回避し、世界で一番安全な「H.264(avc)」コーデックを強制する ▼▼▼
    ydl_opts = {
        'format': 'bestvideo[height<=720][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{song_dir}/video.%(ext)s', 
        'keepvideo': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'noplaylist': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    dl_wav = os.path.join(song_dir, 'video.wav')
    if os.path.exists(dl_wav):
        if os.path.exists(audio_file):
            os.remove(audio_file)
        os.rename(dl_wav, audio_file)
    # ▲▲▲ ここまで ▲▲▲

    print("【3】 音声の特徴量（ボーカルとドラム）をハイブリッド解析中...")
    y, sr = librosa.load(audio_file, sr=None)
    
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(VOCAL_MARGIN, 1.0))

    perc_onsets = librosa.onset.onset_detect(y=y_percussive, sr=sr, pre_max=20, post_max=20, pre_avg=100, delta=0.25)
    perc_times = librosa.frames_to_time(perc_onsets, sr=sr)

    harm_onsets = librosa.onset.onset_detect(y=y_harmonic, sr=sr, pre_max=20, post_max=20, pre_avg=100, delta=0.15)
    harm_times = librosa.frames_to_time(harm_onsets, sr=sr)

    all_times = np.sort(np.concatenate((perc_times, harm_times)))
    filtered_times = []
    for t in all_times:
        if not filtered_times or (t - filtered_times[-1]) >= 0.18:
            filtered_times.append(t)
            
    filtered_times = np.array(filtered_times)
    filtered_frames = librosa.time_to_frames(filtered_times, sr=sr)

    print("【4】 楽曲のテンポ(BPM)とノーツを生成中...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    try:
        bpm = float(tempo[0])
    except (TypeError, IndexError):
        bpm = float(tempo)
        
    bpm_file = os.path.join(song_dir, 'bpm.txt')
    with open(bpm_file, 'w') as f:
        f.write(str(bpm))
    print(f" -> 検出BPM: {bpm:.1f}")

    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    safe_frames = np.minimum(filtered_frames, len(centroids) - 1)
    onset_cents = centroids[safe_frames]

    df = pd.DataFrame({'time': filtered_times, 'cent': onset_cents})
    df['rank'] = df['cent'].rank(method='first')
    df['lane'] = pd.qcut(df['rank'], q=8, labels=False).astype(int)

    for i in range(1, len(df)):
        if df.loc[i, 'lane'] == df.loc[i-1, 'lane']:
            shift = random.choice([-1, 1])
            df.loc[i, 'lane'] = (df.loc[i, 'lane'] + shift) % 8

    df[['time', 'lane']].to_csv(csv_file, index=False)

    print("【5】 サンプラー用クリップ（6種類）を自動抽出中...")
    try:
        S = np.abs(librosa.stft(y_harmonic))
        freqs = librosa.fft_frequencies(sr=sr)
        min_freq_idx = np.argmax(freqs >= FREQ_MIN)
        max_freq_idx = np.argmax(freqs >= FREQ_MAX) if freqs[-1] >= FREQ_MAX else len(freqs) - 1
        high_energy = np.sqrt(np.mean(S[min_freq_idx:max_freq_idx, :]**2, axis=0))
        energy_times = librosa.frames_to_time(np.arange(len(high_energy)), sr=sr)

        total_beats = len(beat_times)
        search_start = min(IGNORE_START_BEATS, total_beats // 4) 
        safe_limit = total_beats - max(EVAL_BEATS, CLIP_BEATS) - IGNORE_END_BEATS - 1
        if safe_limit <= search_start:
            safe_limit = total_beats - max(EVAL_BEATS, CLIP_BEATS) - 1 
            
        search_beat_indices = np.arange(search_start, safe_limit, STEP_BEATS)
        search_times = np.interp(search_beat_indices, np.arange(total_beats), beat_times)
        eval_end_times = np.interp(search_beat_indices + EVAL_BEATS, np.arange(total_beats), beat_times)

        beat_energies = []
        for start_t, end_t in zip(search_times, eval_end_times):
            idx_start = np.argmin(np.abs(energy_times - start_t))
            idx_end = np.argmin(np.abs(energy_times - end_t))
            if idx_end > idx_start:
                avg_energy = np.mean(high_energy[idx_start:idx_end])
            else:
                avg_energy = 0
            beat_energies.append(avg_energy)
        
        beat_energies = np.array(beat_energies)
        selected_indices = []
        block_size = len(search_beat_indices) // NUM_CLIPS 
        
        for i in range(NUM_CLIPS):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < NUM_CLIPS - 1 else len(search_beat_indices)
            block_energies = beat_energies[start_idx:end_idx]
            if len(block_energies) > 0:
                local_max_idx = np.argmax(block_energies)
                global_max_idx = start_idx + local_max_idx
                selected_indices.append(global_max_idx)

        selected_indices.sort()

        for i, global_idx in enumerate(selected_indices):
            start_time = search_times[global_idx]
            end_beat_idx = search_beat_indices[global_idx] + CLIP_BEATS
            end_time = np.interp(end_beat_idx, np.arange(total_beats), beat_times)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            clip_h = y_harmonic[start_sample:end_sample]
            clip_p = y_percussive[start_sample:end_sample]
            clip_y = clip_h + (clip_p * 0.15) 
            
            clip_y_trimmed, _ = librosa.effects.trim(clip_y, top_db=TRIM_DB)
            
            fade_out_time = 0.1 
            fade_samples = int(fade_out_time * sr)
            if len(clip_y_trimmed) > fade_samples:
                fade_curve = np.linspace(1.0, 0.0, fade_samples)
                clip_y_trimmed[-fade_samples:] *= fade_curve

            if len(clip_y_trimmed) > 0:
                clip_y_normalized = librosa.util.normalize(clip_y_trimmed)
            else:
                clip_y_normalized = clip_y_trimmed
            
            clip_filename = os.path.join(song_dir, f"clip_{i+1}.wav")
            sf.write(clip_filename, clip_y_normalized, sr)
            
        print(" -> クリップの抽出が完了しました！")
    except Exception as e:
        print(f" -> [エラー] クリップ抽出中に問題が発生しました: {e}")

    print(f"\n[+] 解析完了！ '{safe_title}' をライブラリに追加しました。\n")
        
# ==========================================
# 2. Launchpad用ユーティリティ
# ==========================================
def get_note_number(x, y):
    if x < 0 or x > 8 or y < 0 or y > 7:
        return None
    return (8 - y) * 10 + (x + 1)

def clear_pad(outport):
    for y in range(8):
        for x in range(9):
            note = get_note_number(x, y)
            if note is not None:
                outport.send(mido.Message('note_on', note=note, velocity=0))

# ==========================================
# 3. ゲームプレイ＆DOPEエンジン
# ==========================================
def play_game(song_dir, song_title):
    audio_file = os.path.join(song_dir, 'audio.wav')
    csv_file = os.path.join(song_dir, 'pattern.csv')
    
    print("\n[+] 譜面と音源を読み込んでいます...")
    df = pd.read_csv(csv_file)
    notes = [{'time': row['time'], 'x': int(row['lane']), 'hit': False, 'missed': False} for _, row in df.iterrows()]

    bpm_file = os.path.join(song_dir, 'bpm.txt')
    if os.path.exists(bpm_file):
        with open(bpm_file, 'r') as f:
            song_bpm = float(f.read().strip())
    else:
        song_bpm = 120.0  
    
    speed_mult = max(0.5, min(song_bpm / 120.0, 2.0)) 
    print(f"[+] 楽曲BPM: {song_bpm:.1f} (ビジュアルスピード: {speed_mult:.2f}倍)")

    cap = None
    vid_fps = 30.0
    if cv2 is not None:
        video_file = None
        for ext in ['mp4', 'webm', 'mkv']:
            path = os.path.join(song_dir, f'video.{ext}')
            if os.path.exists(path):
                video_file = path
                break
        
        if video_file:
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                vid_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[+] 背景動画をロードしました (FPS: {vid_fps:.1f})")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)

    clips = {}
    for i in range(1, 7):
        clip_path = os.path.join(song_dir, f"clip_{i}.wav")
        if os.path.exists(clip_path):
            clips[i] = pygame.mixer.Sound(clip_path)
        else:
            clips[i] = None

    try:
        horn_sound = pygame.mixer.Sound("horn.wav")
        scratch_sound = pygame.mixer.Sound("scratch.wav")
    except FileNotFoundError:
        horn_sound = scratch_sound = None

    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("LAUNCHPAD SUPER DOPE")
    clock = pygame.time.Clock()

    jp_font_path = pygame.font.match_font('notosanscjk,notosanscjkjp,takao,ipagothic,vlgothic,meiryo,msgothic,hiraginosans')
    font_ui = pygame.font.Font(jp_font_path, 50)
    font_large = pygame.font.Font(jp_font_path, 100) 
    
    gui_message = ""
    gui_msg_color = (255, 255, 255)
    gui_msg_time = 0

    print("[+] Launchpad 起動！ (終了は ESC キー)")

    CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
    NUM_RINGS = 40
    RING_SPACING = 60
    TUNNEL_RADIUS = 250
    BASE_SPEED = 10.0 * speed_mult
    ROTATION_SPEED = 0.5 * speed_mult      
    Z_CAMERA = 10.0           
    Z_HIT = 30.0              
    Z_SPAWN = Z_HIT + (NUM_RINGS * RING_SPACING)

    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    FADE_ALPHA = 40

    fx = {
        'flash_color': (0, 0, 0),
        'flash_alpha': 0,
        'flash_mode': pygame.BLEND_RGB_ADD,
        'glitch_shake': 0,
        'particles': []
    }

    rings = []
    time_counter = 0.0

    def get_curve_offset(t):
        x = math.sin(t * 0.7) * 120 + math.cos(t * 0.4) * 80 + math.sin(t * 1.5) * 40
        y = math.cos(t * 0.6) * 100 + math.sin(t * 0.3) * 60 + math.cos(t * 1.2) * 40
        return x, y

    for i in range(NUM_RINGS):
        z = i * RING_SPACING + Z_CAMERA
        tx = time_counter + (i * 0.1)
        x_off, y_off = get_curve_offset(tx)
        rings.append([z, x_off, y_off, (tx * 40) % 360])

    def spawn_explosion(world_x, world_y, z, color):
        fx['flash_color'] = color
        fx['flash_alpha'] = 120
        fx['flash_mode'] = pygame.BLEND_RGB_ADD if random.random() > 0.3 else pygame.BLEND_RGB_SUB
        fx['glitch_shake'] = 60
        
        num_particles = random.randint(30, 50) 
        for _ in range(num_particles):
            vx = random.uniform(-50, 50)
            vy = random.uniform(-50, 50)
            vz = random.uniform(-100, -40) 
            size = random.uniform(10, 40)   
            life = random.uniform(20, 40)  
            sparkle_offset = random.uniform(0, math.pi * 2)
            
            fx['particles'].append({
                'x': world_x, 'y': world_y, 'z': z,
                'vx': vx, 'vy': vy, 'vz': vz,
                'life': life, 'max_life': life,
                'color': color, 'size': size,
                'sparkle_offset': sparkle_offset
            })

    # ====================
    # メインゲームループ
    # ====================
    prev_led_state = {}
    hit_effects = {}
    side_flashes = {} 
    last_vid_surf = None 

    clip_mapping = {89: 1, 79: 2, 69: 3, 59: 4, 49: 5, 39: 6}

    with mido.open_input(PORT_NAME) as inport, \
         mido.open_output(PORT_NAME) as outport:
        
        clear_pad(outport)
        pygame.mixer.music.play()
        start_time = time.time()
        
        score = 0
        combo = 0
        running = True
        screen.fill((0, 0, 0))

        try:
            while running and pygame.mixer.music.get_busy():
                current_time = time.time() - start_time
                
                if cap and cap.isOpened():
                    target_frame_idx = int(current_time * vid_fps)
                    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    read_new_frame = False
                    catch_up_limit = 3  
                    frames_read = 0
                    
                    while current_frame_idx < target_frame_idx and frames_read < catch_up_limit:
                        ret, frame = cap.read()
                        current_frame_idx += 1
                        frames_read += 1
                        if not ret:
                            break
                        read_new_frame = True

                    if read_new_frame and frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
                        last_vid_surf = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                        last_vid_surf.set_alpha(VIDEO_ALPHA) 

                bps = song_bpm / 60.0
                beat_angle = current_time * bps * math.pi * 2

                current_fov = 400 + math.cos(beat_angle) * 120
                current_twist = 0.005 + math.sin(beat_angle / 2.0) * 0.012
                current_tunnel_speed = BASE_SPEED + math.cos(beat_angle) * (BASE_SPEED * 0.6)
                time_counter += (bps * 0.03)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                cw, ch = screen.get_size()
                if cw > 0 and ch > 0:
                    if fade_surface.get_width() != cw or fade_surface.get_height() != ch:
                        fade_surface = pygame.Surface((cw, ch))
                        WIDTH, HEIGHT = cw, ch
                        CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
                        screen.fill((0, 0, 0))

                for msg in inport.iter_pending():
                    if msg.type == 'note_on' and msg.velocity > 0:
                        pressed_note = msg.note
                        
                        if pressed_note in clip_mapping:
                            clip_idx = clip_mapping[pressed_note]
                            if clips[clip_idx]:
                                clips[clip_idx].set_volume(1.0)
                                clips[clip_idx].stop()
                                clips[clip_idx].play()
                                side_flashes[pressed_note] = current_time
                                fx['glitch_shake'] = max(fx['glitch_shake'], 40)
                                
                        elif pressed_note == 19 and horn_sound:
                            horn_sound.set_volume(1.0)
                            horn_sound.play()
                            side_flashes[19] = current_time
                        elif pressed_note == 29 and scratch_sound:
                            scratch_sound.set_volume(1.0)
                            scratch_sound.stop()
                            scratch_sound.play()
                            side_flashes[29] = current_time
                        
                        elif 11 <= pressed_note <= 18:
                            pressed_x = pressed_note - 11
                            hit_found = False
                            
                            for note in notes:
                                if note['x'] == pressed_x and not note['hit'] and not note['missed']:
                                    time_diff = abs(note['time'] - current_time)
                                    if time_diff < 0.2:
                                        note['hit'] = True
                                        hit_found = True
                                        combo += 1
                                        
                                        lane_width = (TUNNEL_RADIUS * 2) / 8
                                        lane_x_off = -TUNNEL_RADIUS + (pressed_x + 0.5) * lane_width
                                        lane_y_off = TUNNEL_RADIUS * 0.5
                                        tx = time_counter + ((Z_HIT - Z_CAMERA) / RING_SPACING) * 0.1
                                        cx, cy = get_curve_offset(tx)
                                        angle = time_counter * ROTATION_SPEED + Z_HIT * current_twist
                                        rx = lane_x_off * math.cos(angle) - lane_y_off * math.sin(angle)
                                        ry = lane_x_off * math.sin(angle) + lane_y_off * math.cos(angle)
                                        world_x, world_y = rx + cx, ry + cy

                                        if time_diff < 0.08:
                                            hit_effects[pressed_x] = {'time': current_time, 'color': 53}
                                            score += 300
                                            gui_message = "PERFECT!!"
                                            gui_msg_color = (0, 255, 255)
                                            spawn_explosion(world_x, world_y, Z_HIT, random.choice([(0, 255, 255), (255, 0, 255)]))
                                        else:
                                            hit_effects[pressed_x] = {'time': current_time, 'color': 13}
                                            score += 100
                                            gui_message = "GREAT"
                                            gui_msg_color = (255, 255, 0)
                                            spawn_explosion(world_x, world_y, Z_HIT, (255, 200, 50))
                                            
                                        gui_msg_time = current_time
                                        break
                            
                            if not hit_found:
                                combo = 0
                                hit_effects[pressed_x] = {'time': current_time, 'color': 5}
                                gui_message = "MISS..."
                                gui_msg_color = (255, 50, 50)
                                gui_msg_time = current_time

                current_led_state = {}
                bg_palette = [41, 45, 49, 45, 41] 
                for y in range(7):
                    for x in range(8):
                        color_idx = int(current_time * 1.5 + x * 0.2 + y * 0.2) % len(bg_palette)
                        current_led_state[get_note_number(x, y)] = bg_palette[color_idx]

                for flash_x in list(hit_effects.keys()):
                    if current_time - hit_effects[flash_x]['time'] < 0.15:
                        flash_color_led = hit_effects[flash_x]['color']
                        if flash_color_led != 5:
                            for y in range(7):
                                current_led_state[get_note_number(flash_x, y)] = flash_color_led

                for note in notes:
                    if note['hit'] or note['missed']:
                        continue
                    time_diff = note['time'] - current_time
                    if time_diff < -0.2:
                        note['missed'] = True
                        combo = 0
                        gui_message = "MISS..."
                        gui_msg_color = (255, 50, 50)
                        gui_msg_time = current_time
                    elif time_diff <= FALL_TIME:
                        y = int(7 - (time_diff / FALL_TIME) * 7)
                        if 0 <= y <= 7:
                            current_led_state[get_note_number(note['x'], y)] = 5
                
                for x in range(8):
                    num = get_note_number(x, 7)
                    if x in hit_effects:
                        if current_time - hit_effects[x]['time'] < 0.15:
                            current_led_state[num] = hit_effects[x]['color']
                        else:
                            del hit_effects[x]
                            current_led_state[num] = 3
                    else:
                        current_led_state[num] = 3

                for note, idx in clip_mapping.items():
                    if clips[idx]:
                        if note in side_flashes and current_time - side_flashes[note] < 0.15:
                            current_led_state[note] = 3  
                        else:
                            current_led_state[note] = 53 
                    else:
                        current_led_state[note] = 0

                if horn_sound:
                    current_led_state[19] = 3 if (19 in side_flashes and current_time - side_flashes[19] < 0.15) else 21
                if scratch_sound:
                    current_led_state[29] = 3 if (29 in side_flashes and current_time - side_flashes[29] < 0.15) else 13

                for note_num in prev_led_state:
                    if note_num not in current_led_state:
                        outport.send(mido.Message('note_on', note=note_num, velocity=0))
                for note_num, color in current_led_state.items():
                    if prev_led_state.get(note_num) != color:
                        outport.send(mido.Message('note_on', note=note_num, velocity=color))
                prev_led_state = current_led_state

                eff_cx = CENTER_X + (random.randint(-fx['glitch_shake'], fx['glitch_shake']) if fx['glitch_shake'] > 0 else 0)
                eff_cy = CENTER_Y + (random.randint(-fx['glitch_shake'], fx['glitch_shake']) if fx['glitch_shake'] > 0 else 0)
                if fx['glitch_shake'] > 0: fx['glitch_shake'] -= 5

                def project(x, y, z):
                    z_safe = max(z, 0.1) 
                    factor = current_fov / z_safe
                    px = int(x * factor + eff_cx)
                    py = int(y * factor + eff_cy)
                    return px, py

                current_hue = (time_counter * 50) % 360
                bg_color = pygame.Color(0)
                bg_color.hsva = (current_hue, 100, 20, 100) 
                fade_surface.fill(bg_color)
                fade_surface.set_alpha(FADE_ALPHA)
                screen.blit(fade_surface, (0, 0))

                if last_vid_surf is not None:
                    screen.blit(last_vid_surf, (0, 0))

                if fx['flash_alpha'] > 0:
                    flash_surf = pygame.Surface((WIDTH, HEIGHT))
                    flash_surf.fill(fx['flash_color'])
                    flash_surf.set_alpha(int(fx['flash_alpha']))
                    screen.blit(flash_surf, (0, 0), special_flags=fx['flash_mode'])
                    fx['flash_alpha'] = max(0, fx['flash_alpha'] - 30) 

                for i in range(len(rings)):
                    rings[i][0] -= current_tunnel_speed
                    rings[i][3] = (rings[i][3] + 2) % 360 

                if len(rings) > 0 and rings[0][0] <= Z_CAMERA:
                    rings.pop(0)
                    last_z = rings[-1][0]
                    tx = time_counter + (NUM_RINGS * 0.1)
                    x_off, y_off = get_curve_offset(tx)
                    rings.append([last_z + RING_SPACING, x_off, y_off, (tx * 40) % 360])

                # ▼▼▼ 修正：動画が再生されていない場合のみ、ワイヤーフレームのトンネルを描画する ▼▼▼
                if last_vid_surf is None:
                    for i in range(len(rings) - 1, 0, -1):
                        z1, x1, y1, hue1 = rings[i]
                        z2, x2, y2, hue2 = rings[i-1]
                        color = pygame.Color(0)
                        color.hsva = (hue1, 100, 100, 100)

                        angle1 = time_counter * ROTATION_SPEED + z1 * current_twist
                        angle2 = time_counter * ROTATION_SPEED + z2 * current_twist

                        def get_rotated_corners(angle):
                            corners = []
                            for cx, cy in [(-TUNNEL_RADIUS, -TUNNEL_RADIUS),
                                           ( TUNNEL_RADIUS, -TUNNEL_RADIUS),
                                           ( TUNNEL_RADIUS,  TUNNEL_RADIUS),
                                           (-TUNNEL_RADIUS,  TUNNEL_RADIUS)]:
                                rx = cx * math.cos(angle) - cy * math.sin(angle)
                                ry = cx * math.sin(angle) + cy * math.cos(angle)
                                corners.append((rx, ry))
                            return corners

                        rot_corners1 = get_rotated_corners(angle1)
                        rot_corners2 = get_rotated_corners(angle2)

                        proj1 = []
                        proj2 = []
                        for j in range(4):
                            cx1, cy1 = rot_corners1[j]
                            cx2, cy2 = rot_corners2[j]
                            proj1.append(project(cx1 + x1, cy1 + y1, z1))
                            proj2.append(project(cx2 + x2, cy2 + y2, z2))

                        pygame.draw.polygon(screen, color, proj1, 4)
                        for j in range(4):
                            pygame.draw.line(screen, color, proj1[j], proj2[j], 4)
                # ▲▲▲ ここまで ▲▲▲

                for note in notes:
                    if note['hit'] or note['missed']:
                        continue
                        
                    time_diff = note['time'] - current_time
                    if -0.1 < time_diff <= FALL_TIME:
                        progress = time_diff / FALL_TIME
                        z = Z_HIT + progress * (Z_SPAWN - Z_HIT)
                        lane_width = (TUNNEL_RADIUS * 2) / 8
                        lane_x_off = -TUNNEL_RADIUS + (note['x'] + 0.5) * lane_width
                        lane_y_off = TUNNEL_RADIUS * 0.5
                        tx = time_counter + ((z - Z_CAMERA) / RING_SPACING) * 0.1
                        cx, cy = get_curve_offset(tx)
                        angle = time_counter * ROTATION_SPEED + z * current_twist
                        rx = lane_x_off * math.cos(angle) - lane_y_off * math.sin(angle)
                        ry = lane_x_off * math.sin(angle) + lane_y_off * math.cos(angle)
                        px, py = project(rx + cx, ry + cy, z)

                        base_size = max(10, int((current_fov / max(z, 0.1)) * 40))
                        intensity = min(255, max(100, int(255 * (1.2 - progress))))
                        
                        halo_color = (intensity, 50, 200)
                        outer_color = (255, max(50, intensity - 50), max(50, intensity - 50))
                        
                        pygame.draw.circle(screen, halo_color, (px, py), base_size + 20)
                        pygame.draw.circle(screen, outer_color, (px, py), base_size)
                        pygame.draw.circle(screen, (255,255,255), (px, py), max(2, int(base_size * 0.5)))

                for p in fx['particles'][:]:
                    p['life'] -= 1
                    if p['life'] <= 0:
                        fx['particles'].remove(p)
                        continue

                    p['x'] += p['vx']
                    p['y'] += p['vy']
                    p['z'] -= p['vz']
                    p['vy'] += 1.5 

                    if p['z'] > 0.1:
                        px, py = project(p['x'], p['y'], p['z'])
                        current_size = max(0.1, p['size'] * (p['life'] / p['max_life']))
                        proj_size = max(1, int((current_fov / p['z']) * current_size))
                        
                        sparkle_factor = (math.sin(p['life'] * 1.5 + p['sparkle_offset']) + 1) / 2.0
                        
                        pygame.draw.circle(screen, p['color'], (px, py), proj_size)
                        if sparkle_factor > 0.5:
                            core_size = max(1, int(proj_size * 0.7))
                            pygame.draw.circle(screen, (255, 255, 255), (px, py), core_size)

                if fx['glitch_shake'] > 50:
                    offset_x = random.randint(-15, 15)
                    offset_y = random.randint(-15, 15)
                    screen.blit(screen, (offset_x, offset_y), special_flags=pygame.BLEND_RGB_ADD)

                title_text = font_ui.render(song_title, True, (200, 200, 200))
                score_text = font_ui.render(f"SCORE: {score}", True, (200, 200, 200))
                combo_text = font_ui.render(f"COMBO: {combo}", True, (255, 200, 50))
                
                screen.blit(title_text, (40, 40))
                screen.blit(score_text, (40, 100))  
                screen.blit(combo_text, (40, 160))

                if current_time - gui_msg_time < 0.4:
                    msg_text = font_large.render(gui_message, True, gui_msg_color)
                    msg_text = pygame.transform.rotate(msg_text, 10)
                    screen.blit(msg_text, (40, 220))

                pygame.display.flip()
                clock.tick(60)

        except KeyboardInterrupt:
            pass
        finally:
            pygame.mixer.music.stop()
            pygame.quit()
            clear_pad(outport)
            if cap:
                cap.release()
            print(f"\n=== GAME OVER ===")
            print(f"Final Score: {score}")

# ==========================================
# メインメニュー
# ==========================================
def main():
    while True:
        os.makedirs(SONGS_DIR, exist_ok=True)
        songs = [d for d in os.listdir(SONGS_DIR) if os.path.isdir(os.path.join(SONGS_DIR, d))]
        
        print("\n" + "="*40)
        print(" LAUNCHPAD SUPER DOPE - Main Menu")
        print("="*40)
        print(" [0] 新しい曲をYouTubeから追加する")
        
        for i, song in enumerate(songs, 1):
            print(f" [{i}] プレイ: {song}")
        print(" [q] 終了")
        print("="*40)
        
        choice = input("番号を選択してください: ").strip()
        
        if choice.lower() == 'q':
            break
        elif choice == '0':
            download_and_analyze()
        elif choice.isdigit() and 1 <= int(choice) <= len(songs):
            song_title = songs[int(choice)-1]
            song_dir = os.path.join(SONGS_DIR, song_title)
            play_game(song_dir, song_title)
        else:
            print("[!] 無効な入力です。")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了します。")
