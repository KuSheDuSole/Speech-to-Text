import torch
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference import EncoderClassifier
from sklearn.metrics.pairwise import cosine_distances
from config import SILERO_DIR, MONOLOGUE_STD_THRESHOLD
import os

print("🔊 Загружаем Silero VAD...")
model_vad, utils = torch.hub.load(
    repo_or_dir=str(SILERO_DIR),
    model='silero_vad',
    source='local',
    force_reload=False
)
(get_speech_ts, _, _, _, _) = utils


print("🧬 Загружаем ECAPA-TDNN...")
device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)


def extract_embedding(audio_segment):
    signal = torch.tensor(audio_segment).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = classifier.encode_batch(signal).cpu().numpy()
    emb = np.squeeze(emb)
    if emb.ndim == 2:
        emb = np.mean(emb, axis=0)
    return emb


def diarize(audio_path, min_speakers=1, max_speakers=4,
            min_segment_len_s=0.25, short_segment_thresh_s=1.0,
            small_speaker_total_thresh_s=0.7, cluster_merge_cosine_thresh=0.12):

    def merge_adjacent(segs, gap_threshold=0.30):
        if not segs:
            return []
        merged = [segs[0].copy()]
        for s in segs[1:]:
            last = merged[-1]
            if s["speaker"] == last["speaker"] and s["start"] - last["end"] <= gap_threshold:
                last["end"] = s["end"]
            else:
                merged.append(s.copy())
        return merged

    wav, sr = librosa.load(audio_path, sr=16000)
    wav_tensor = torch.tensor(wav)

    speech_ts = get_speech_ts(wav_tensor, model_vad)
    if not speech_ts:
        raise RuntimeError("No speech detected.")

    raw_segments = []
    for seg in speech_ts:
        start_s = seg["start"] / sr
        end_s = seg["end"] / sr
        if end_s - start_s < min_segment_len_s:
            continue
        raw_segments.append({"start": start_s, "end": end_s})

    if not raw_segments:
        raise RuntimeError("No segments left after VAD filtering.")

    embeddings = []
    for seg in raw_segments:
        s_idx = int(seg["start"] * sr)
        e_idx = int(seg["end"] * sr)
        audio_seg = wav[s_idx:e_idx]
        emb = extract_embedding(audio_seg)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    if embeddings.shape[0] == 1:
        raw_segments[0]["speaker"] = "speaker_0"
        return raw_segments

    dist_mtx = cosine_distances(embeddings, embeddings)
    upper = dist_mtx[np.triu_indices(dist_mtx.shape[0], k=1)]

    mean_dist = np.mean(upper)
    std_dist = np.std(upper)

    print(f"\n🔍 cosine mean={mean_dist:.4f}, std={std_dist:.4f}")


    if (std_dist < MONOLOGUE_STD_THRESHOLD) or (max_speakers == 1):
        print("🎙 Detected MONOLOGUE (low embedding variance). Skipping clustering.")
        for seg in raw_segments:
            seg["speaker"] = "speaker_0"
        raw_segments = merge_adjacent(raw_segments, gap_threshold=0.30)
        return raw_segments

    best_k = 2
    best_score = -1.0
    scores = {}
    print("\n📈 selecting number of speakers (silhouette)...")

    for k in range(2, max(2, max_speakers) + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k).fit(embeddings)
            labels = model.labels_
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            scores[k] = score
            print(f"  k={k}: silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    if best_score < 0.10:
        print("⚠ silhouette low for all k — falling back to 2 clusters.")
        best_k = 2
    print(f"→ chosen k = {best_k} (score={best_score:.4f})")

    clustering = AgglomerativeClustering(n_clusters=best_k).fit(embeddings)
    labels = clustering.labels_

    centroids = {}
    for i, lab in enumerate(labels):
        centroids.setdefault(lab, []).append(embeddings[i])
    for lab in list(centroids.keys()):
        centroids[lab] = np.mean(centroids[lab], axis=0)
        centroids[lab] = centroids[lab] / (np.linalg.norm(centroids[lab]) + 1e-8)

    labs = sorted(centroids.keys())
    merged_map = {}
    used = set()
    next_lab = 0

    for i, la in enumerate(labs):
        if la in used:
            continue
        merged_map[la] = next_lab
        used.add(la)
        for lb in labs[i + 1:]:
            if lb in used:
                continue
            cos_dist = 1.0 - np.dot(centroids[la], centroids[lb])
            if cos_dist < cluster_merge_cosine_thresh:
                merged_map[lb] = next_lab
                used.add(lb)
        next_lab += 1

    remapped = [merged_map[l] for l in labels]
    for i, seg in enumerate(raw_segments):
        seg["speaker"] = f"speaker_{remapped[i]}"

    merged_segments = merge_adjacent(raw_segments, gap_threshold=0.30)

    totals = {}
    for seg in merged_segments:
        totals.setdefault(seg["speaker"], 0.0)
        totals[seg["speaker"]] += seg["end"] - seg["start"]

    tiny_speakers = [sp for sp, t in totals.items() if t < small_speaker_total_thresh_s]
    if tiny_speakers:
        print(f"🧹 tiny speakers: {tiny_speakers}. Reassigning.")
        for seg in merged_segments:
            if seg["speaker"] in tiny_speakers:
                for i, s in enumerate(merged_segments):
                    if s is seg:
                        idx = i
                        break
                prev_idx = idx - 1 if idx > 0 else None
                next_idx = idx + 1 if idx < len(merged_segments) - 1 else None
                if prev_idx is not None and next_idx is not None:
                    gap_prev = seg["start"] - merged_segments[prev_idx]["end"]
                    gap_next = merged_segments[next_idx]["start"] - seg["end"]
                    seg["speaker"] = merged_segments[prev_idx]["speaker"] if gap_prev <= gap_next else merged_segments[next_idx]["speaker"]
                elif prev_idx is not None:
                    seg["speaker"] = merged_segments[prev_idx]["speaker"]
                elif next_idx is not None:
                    seg["speaker"] = merged_segments[next_idx]["speaker"]
                else:
                    seg["speaker"] = "speaker_0"
        merged_segments = merge_adjacent(merged_segments, gap_threshold=0.30)

    fixed = []
    for i, seg in enumerate(merged_segments):
        dur = seg["end"] - seg["start"]
        if dur < short_segment_thresh_s and 0 < i < len(merged_segments) - 1:
            prev = merged_segments[i - 1]
            nxt  = merged_segments[i + 1]
            if prev["speaker"] == nxt["speaker"]:
                seg["speaker"] = prev["speaker"]
            else:
                sp_prev = prev["speaker"]
                sp_next = nxt["speaker"]
                speaker_centroids = {f"speaker_{merged_map.get(lab, lab)}": cent
                                     for lab, cent in centroids.items()}
                c_prev = speaker_centroids.get(sp_prev)
                c_next = speaker_centroids.get(sp_next)
                if c_prev is not None and c_next is not None:
                    mid = (seg["start"] + seg["end"]) / 2.0
                    centers = [(r["start"] + r["end"]) / 2.0 for r in raw_segments]
                    nearest_idx = int(np.argmin(np.abs(np.array(centers) - mid)))
                    emb_vec = embeddings[nearest_idx]
                    seg["speaker"] = sp_prev if (1.0 - np.dot(emb_vec, c_prev)) <= (1.0 - np.dot(emb_vec, c_next)) else sp_next
                else:
                    seg["speaker"] = prev["speaker"]
        fixed.append(seg)
    merged_segments = merge_adjacent(fixed, gap_threshold=0.30)

    normalized = []
    map_sp = {}
    counter = 0
    for seg in merged_segments:
        sp = seg["speaker"]
        if sp not in map_sp:
            map_sp[sp] = f"speaker_{counter}"
            counter += 1
        seg_copy = seg.copy()
        seg_copy["speaker"] = map_sp[sp]
        normalized.append(seg_copy)

    return merge_adjacent(normalized, gap_threshold=0.30)


def merge_adjacent_segments(segments, gap_threshold=0.30):
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= gap_threshold:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged