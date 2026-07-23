"""
NOT: Bu proje otomatik al-sat ajanı oluşturmadığı ve öğrenme mekanizması
olarak skor-bazlı ağırlıklandırma + genetik algoritma (bkz. feedback_loop.py,
genetic_optimizer.py) kullandığı için Stable-Baselines3/PPO eğitimi şu an
AKTİF DEĞİLDİR. Streamlit Cloud'un ücretsiz katmanında RL eğitimi pratik
değildir (CPU/RAM/süre kısıtları).

Bu dosya, ileride ayrı, kalıcı hesaplama kaynağı olan bir ortamda (örn. yerel
makine, dedicated CI runner) RL denenmek istenirse iskelet olarak bırakılmıştır.
"""

from __future__ import annotations


def train_ppo_agent(*args, **kwargs):
    raise NotImplementedError(
        "PPO eğitimi bu proje kapsamında aktif değil. Bkz. "
        "src/learning/feedback_loop.py ve src/learning/genetic_optimizer.py."
    )


if __name__ == "__main__":
    print(
        "PPO eğitimi devre dışı. Bunun yerine şunu çalıştırın:\n"
        "  python scripts/train_model.py"
    )
