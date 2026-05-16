from __future__ import annotations
import re
import numpy as np
from typing import Optional

FILLER_WORDS = {
    "эм","э","ну","вот","типа","как бы","короче","значит",
    "понимаешь","слушай","блин","собственно","вообще","кстати",
    "то есть","в общем","ладно","окей","ок",
}


def _extract_pitch(audio: np.ndarray, sr: int):
    try:
        import librosa
        f0, voiced, _ = librosa.pyin(audio.astype(np.float32),
                                      fmin=librosa.note_to_hz("C2"),
                                      fmax=librosa.note_to_hz("C7"),
                                      sr=sr, frame_length=2048, hop_length=512)
        return f0, voiced
    except Exception:
        return np.array([]), np.array([], dtype=bool)


def _jitter(f0: np.ndarray, voiced) -> float:
    if f0 is None or voiced is None or len(f0) == 0:
        return 0.0
    vf = f0[voiced & ~np.isnan(f0)] if len(voiced)==len(f0) else f0[~np.isnan(f0)]
    if len(vf) < 4:
        return 0.0
    periods = 1.0 / (vf + 1e-9)
    return round(float(np.clip(
        np.mean(np.abs(np.diff(periods))) / (np.mean(periods)+1e-9) * 100, 0, 20)), 2)


def _shimmer(audio: np.ndarray, sr: int, voiced, hop: int = 512) -> float:
    if voiced is None or len(voiced) == 0:
        return 0.0
    fl = 2048
    amps = []
    for i, v in enumerate(voiced):
        if not v:
            continue
        s, e = i*hop, i*hop+fl
        if e > len(audio):
            break
        rms = float(np.sqrt(np.mean(audio[s:e]**2)))
        amps.append(rms)
    if len(amps) < 4:
        return 0.0
    a = np.array(amps)
    return round(float(np.clip(
        np.mean(np.abs(np.diff(a))) / (np.mean(a)+1e-9) * 100, 0, 30)), 2)


def _active_rms(audio: np.ndarray, sr: int) -> list[float]:
    fs = int(0.02*sr)
    vals = [float(np.sqrt(np.mean(audio[i:i+fs]**2)))
            for i in range(0, len(audio)-fs, fs)]
    if not vals:
        return []
    thr = np.percentile(vals, 30)
    return [v for v in vals if v > thr]


def _ttr(text: str) -> float:
    w = re.findall(r'[а-яёa-z]+', text.lower())
    return round(len(set(w))/len(w), 3) if w else 0.0


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)


def analyze_speech(segments: list[dict], full_audio: np.ndarray,
                   sample_rate: int,
                   logits_per_segment: Optional[list] = None) -> dict:
    if not segments:
        return {}

    SR  = sample_rate
    tot = len(full_audio) / SR
    ga  = _active_rms(full_audio, SR)
    gmax = max(ga) if ga else 1.0

    segs_s = sorted(segments, key=lambda x: x["start"])
    sp_dur = sum(s["end"]-s["start"] for s in segs_s)
    pauses = [segs_s[i]["start"]-segs_s[i-1]["end"]
              for i in range(1,len(segs_s))
              if segs_s[i]["start"]-segs_s[i-1]["end"] > 0.25]

    spk: dict = {}
    for seg in segs_s:
        sp = seg.get("speaker","speaker_0")
        if sp not in spk:
            spk[sp] = {"segs":[],"dur":0.0,"words":[],"fillers":0,"repeats":0}
        d = spk[sp]
        d["segs"].append(seg); d["dur"] += seg["end"]-seg["start"]
        txt = seg.get("text","")
        if txt:
            ws = txt.lower().split(); d["words"].extend(ws)
            for fw in FILLER_WORDS:
                d["fillers"] += txt.lower().count(fw)
            for i in range(1,len(ws)):
                if ws[i]==ws[i-1] and len(ws[i])>1:
                    d["repeats"] += 1

    speaker_stats = {}
    for sp, d in spk.items():
        dur = d["dur"]; wc = len(d["words"]); text = " ".join(d["words"])
        sdur = max(dur, wc*0.35, 1.0)
        wpm  = wc/sdur*60
        if wpm<80:    pace="очень медленный"
        elif wpm<120: pace="медленный"
        elif wpm<160: pace="нормальный"
        elif wpm<200: pace="быстрый"
        else:         pace="очень быстрый"

        fp = d["fillers"]/wc*100 if wc else 0.0
        rp = d["repeats"]/wc*100 if wc else 0.0

        chunks = [full_audio[int(s["start"]*SR):int(s["end"]*SR)]
                  for s in d["segs"] if int(s["end"]*SR)>int(s["start"]*SR)]
        combined = np.concatenate(chunks) if chunks else np.array([])

        act = _active_rms(combined, SR) if len(combined) else []
        en  = round(float(np.clip(np.mean(act)/gmax,0,1)),3) if act else 0.0
        ecv = round(float(np.std(act)/(np.mean(act)+1e-9)),3) if len(act)>2 else 0.0
        if ecv<0.3:   stab="ровный голос"
        elif ecv<0.6: stab="умеренно переменчивый"
        else:         stab="много скачков громкости"

        if len(combined)>SR*0.5:
            f0, voiced = _extract_pitch(combined, SR)
        else:
            f0, voiced = np.array([]), np.array([], dtype=bool)

        vf0 = f0[(voiced & ~np.isnan(f0))] if (len(voiced)==len(f0) and len(f0)>0) else np.array([])
        if len(vf0)>4:
            pm  = round(float(np.mean(vf0)),1)
            pr  = round(float(np.max(vf0)-np.min(vf0)),1)
            pcv = float(np.std(vf0)/(np.mean(vf0)+1e-9))
            if pcv<0.08:   inton="монотонная (мало интонационных перепадов)"
            elif pcv<0.18: inton="умеренная"
            else:          inton="выразительная (широкий диапазон)"
        else:
            pm=pr=0.0; inton="не определено"

        jit = _jitter(f0, voiced)
        shim = _shimmer(combined, SR, voiced) if len(combined) else 0.0
        ttr = _ttr(text)
        sents = re.split(r'[.!?]+', text)
        asl   = round(float(np.mean([len(s.split()) for s in sents if s.strip()])),1) if sents else 0.0

        speaker_stats[sp] = {
            "dur_sec":round(dur,1),"word_count":wc,"wpm":round(wpm,1),"pace":pace,
            "filler_count":d["fillers"],"filler_pct":round(fp,1),
            "repeat_count":d["repeats"],"repeat_pct":round(rp,1),
            "energy_norm":en,"stability":stab,
            "pitch_mean":pm,"pitch_range":pr,"intonation":inton,
            "jitter_pct":jit,"shimmer_pct":shim,"ttr":ttr,"avg_sent_len":asl,
        }

    asr_conf = None
    if logits_per_segment:
        ents = []
        for lg in logits_per_segment:
            if lg is None: continue
            p = _softmax(lg)
            ents.append(float(np.mean(-np.sum(p*np.log(p+1e-9), axis=-1))))
        if ents:
            vsz = logits_per_segment[0].shape[-1] if logits_per_segment[0] is not None else 32
            asr_conf = round(1.0-float(np.mean(ents))/np.log(vsz),3)

    return {
        "total_sec":   round(tot,1),
        "speech_sec":  round(sp_dur,1),
        "silence_pct": round((1.0-sp_dur/tot)*100,1) if tot>0 else 0.0,
        "pause_count": len(pauses),
        "avg_pause":   round(float(np.mean(pauses)),2) if pauses else 0.0,
        "max_pause":   round(float(np.max(pauses)),2) if pauses else 0.0,
        "speakers":    speaker_stats,
        **({"asr_confidence":asr_conf} if asr_conf is not None else {}),
    }


def format_analysis(analysis: dict) -> str:
    if not analysis:
        return ""
    L = ["\n📊 АНАЛИЗ РЕЧИ:\n"+"─"*56]
    L.append(f"⏱  Длительность записи:     {analysis['total_sec']} сек")
    L.append(f"🗣  Из них речь:             {analysis['speech_sec']} сек")
    L.append(f"🔇 Тишина и паузы:          {analysis['silence_pct']}%")
    if analysis["pause_count"]>0:
        L.append(f"⏸  Паузы:                   {analysis['pause_count']} шт  "
                 f"│  средняя {analysis['avg_pause']}с  │  макс {analysis['max_pause']}с")
    else:
        L.append("⏸  Паузы между фразами:     не выявлены")
    if "asr_confidence" in analysis:
        c=analysis["asr_confidence"]
        bar="█"*int(c*20)+"░"*(20-int(c*20))
        lbl="высокая" if c>0.85 else ("средняя" if c>0.65 else "низкая")
        L.append(f"🎯 Чёткость речи*:          [{bar}] {c:.1%} ({lbl})")
        L.append(f"   * уверенность модели в распознавании звуков")
    for sp,s in analysis["speakers"].items():
        L.append(f"\n👤 {sp.upper()}:")
        L.append(f"   Темп речи:          {s['wpm']} сл/мин — {s['pace']}")
        L.append(f"                       (норма русской речи: 120–180 сл/мин)")
        L.append(f"   Слов произнесено:   {s['word_count']}")
        if s["filler_count"]>0:
            tag="⚠️ много" if s["filler_pct"]>10 else "норма"
            L.append(f"   Слова-паразиты:     {s['filler_count']} шт ({s['filler_pct']}%) — {tag}")
        else:
            L.append("   Слова-паразиты:     не обнаружены ✅")
        if s["repeat_count"]>0:
            L.append(f"   Запинки/повторы:    {s['repeat_count']} шт ({s['repeat_pct']}%)")
        else:
            L.append("   Запинки/повторы:    не обнаружены ✅")
        ttr=s["ttr"]
        tl="богатый словарь" if ttr>0.85 else ("средний" if ttr>0.65 else "ограниченный / много повторений")
        L.append(f"   Словарное богатство: {ttr:.2f} — {tl}  (TTR)")
        L.append(f"   Дл. предложений:    {s['avg_sent_len']} слов в среднем")
        if s["pitch_mean"]>0:
            L.append(f"   Высота голоса:      {s['pitch_mean']} Гц  (мужской 85–180, женский 165–255)")
            L.append(f"   Интонация:          {s['intonation']}  (диапазон {s['pitch_range']} Гц)")
        else:
            L.append("   Высота голоса:      не удалось определить")
        j=s["jitter_pct"]
        jl="норма ✅" if j<1.0 else ("небольшая нестабильность" if j<2.0 else "⚠️ заметная нестабильность (стресс/усталость)")
        L.append(f"   Дрожание голоса:    {j}% — {jl}  (Jitter, норма <1%)")
        sh=s["shimmer_pct"]
        sl="норма ✅" if sh<3.0 else ("небольшая" if sh<5.0 else "⚠️ заметная (хриплость/усталость)")
        L.append(f"   Нестаб. громкости:  {sh}% — {sl}  (Shimmer, норма <3%)")
        L.append(f"   Ровность голоса:    {s['stability']}")
        L.append(f"   Громкость (норм.):  {s['energy_norm']}  (0=тихо, 1=макс в записи)")
    L.append("─"*56)
    return "\n".join(L)