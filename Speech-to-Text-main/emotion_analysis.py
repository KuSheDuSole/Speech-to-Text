from __future__ import annotations
import numpy as np
import torch
from typing import Optional

MODEL_ID = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"

EMOTION_RU = {
    "neutral":    ("нейтральный",        "спокойная ровная речь"),
    "angry":      ("злость/раздражение", "напряжённая речь, возможен конфликт"),
    "positive":   ("позитивный",         "приподнятое настроение"),
    "sad":        ("грусть",             "подавленное настроение, усталость"),
    "enthusiasm": ("воодушевление",      "высокая вовлечённость, энергичность"),
    "fear":       ("тревога",            "напряжение, неуверенность"),
    "disgust":    ("недовольство",       "негативная оценка"),
    "other":      ("другое",             "неопределённая эмоция"),
}

ID2LABEL = {0:"neutral", 1:"angry", 2:"positive", 3:"sad",
            4:"enthusiasm", 5:"fear", 6:"disgust", 7:"other"}


class EmotionAnalyzer:
    def __init__(self, device: Optional[str] = None, model_id: str = MODEL_ID):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"🎭 Загружаем модель эмоций ({model_id})...")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model     = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.to(device).eval()
        cfg            = self.model.config
        self.id2label  = getattr(cfg, "id2label", ID2LABEL)
        print(f"✅ Модель эмоций загружена на {device}")

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        if len(audio) == 0:
            return {"label":"other","ru_label":"не определено","description":"","confidence":0.0,"scores":{}}
        target_sr = self.extractor.sampling_rate
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except Exception:
                pass
        try:
            inputs = self.extractor(audio.astype(np.float32), sampling_rate=target_sr,
                                    return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
            probs     = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            top_idx   = int(np.argmax(probs))
            label     = self.id2label.get(top_idx, "other")
            ru, desc  = EMOTION_RU.get(label, (label, ""))
            scores    = {self.id2label.get(i,f"cls_{i}"): round(float(p),3)
                         for i,p in enumerate(probs)}
            return {"label":label,"ru_label":ru,"description":desc,
                    "confidence":round(float(probs[top_idx]),3),"scores":scores}
        except Exception:
            return {"label":"other","ru_label":"не определено","description":"","confidence":0.0,"scores":{}}

    def analyze_segments(self, segments: list[dict], full_audio: np.ndarray,
                         sample_rate: int) -> dict:
        if not segments:
            return {}
        SR = sample_rate
        spk_preds: dict[str,list] = {}
        for seg in segments:
            if seg["end"] - seg["start"] < 0.5:
                continue
            sp  = seg.get("speaker","speaker_0")
            s   = max(0, int(seg["start"]*SR))
            e   = min(len(full_audio), int(seg["end"]*SR))
            res = self.predict(full_audio[s:e], SR)
            res.update({"start":seg["start"],"end":seg["end"],"text":seg.get("text","")})
            spk_preds.setdefault(sp,[]).append(res)

        stats = {}
        for sp, preds in spk_preds.items():
            weights: dict[str,float] = {}
            for p in preds:
                weights[p["label"]] = weights.get(p["label"],0.0) + p["confidence"]
            dom = max(weights, key=weights.get)
            dom_ru, dom_desc = EMOTION_RU.get(dom,(dom,""))
            unique = len(set(p["label"] for p in preds))
            if unique == 1:   emo_range = "однородная эмоциональная окраска"
            elif unique <= 3: emo_range = "умеренный эмоциональный диапазон"
            else:             emo_range = "широкий эмоциональный диапазон"
            stats[sp] = {
                "dominant_emotion": dom_ru,
                "dominant_desc":    dom_desc,
                "emotion_range":    emo_range,
                "unique_emotions":  unique,
                "avg_confidence":   round(float(np.mean([p["confidence"] for p in preds])),3),
                "timeline": [{"start":p["start"],"end":p["end"],
                              "emotion":p["ru_label"],"conf":p["confidence"]} for p in preds],
                "label_weights": {EMOTION_RU.get(k,(k,""))[0]: round(v,2)
                                  for k,v in weights.items()},
            }
        return stats


def format_emotion(stats: dict) -> str:
    if not stats:
        return ""
    lines = ["\n🎭 АНАЛИЗ ЭМОЦИЙ:\n" + "─"*54]
    lines.append("   Модель: HuBERT Large (DUSHA, русский язык, Apache 2.0)")
    lines.append("   ⚠️  Результат вероятностный\n")
    for sp, s in stats.items():
        lines.append(f"👤 {sp.upper()}:")
        lines.append(f"   Доминирующая эмоция:  {s['dominant_emotion']}")
        lines.append(f"                         {s['dominant_desc']}")
        lines.append(f"   Эмоц. диапазон:       {s['emotion_range']}")
        lines.append(f"   Уверенность модели:   {s['avg_confidence']:.1%}")
        if s["label_weights"]:
            lines.append("   Распределение:")
            for emo, w in sorted(s["label_weights"].items(), key=lambda x:-x[1]):
                bar = "▓" * int(w*4)
                lines.append(f"     {emo:<24} {bar} {w:.2f}")
        if s["timeline"]:
            lines.append("   По сегментам:")
            for item in s["timeline"][:6]:
                t = f"{item['start']:.1f}–{item['end']:.1f}с"
                lines.append(f"     [{t:>12}]  {item['emotion']}  ({item['conf']:.0%})")
            if len(s["timeline"]) > 6:
                lines.append(f"     ... ещё {len(s['timeline'])-6} сегментов")
        lines.append("")
    lines.append("─"*54)
    return "\n".join(lines)