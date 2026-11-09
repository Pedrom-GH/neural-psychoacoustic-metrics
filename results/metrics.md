| Split | Model | Loudness MAE [phon] | Loudness r | Sharpness MAE [acum] | Sharpness r |
|---|---|---|---|---|---|
| test | MetricNet | 0.43 | 1.000 | 0.030 | 0.999 |
| test | Linear on band levels | 10.40 | 0.731 | 0.586 | 0.640 |
| test | Linear on overall SPL | 6.41 | 0.922 | 1.022 | 0.027 |
| ood | MetricNet | 1.10 | 0.997 | 0.028 | 1.000 |
| ood | Linear on band levels | 15.54 | 0.360 | 1.000 | 0.638 |
| ood | Linear on overall SPL | 5.87 | 0.989 | 1.262 | -0.012 |

| Family | n | Loudness MAE [phon] | Sharpness MAE [acum] |
|---|---|---|---|
| bandpass_noise | 250 | 0.48 | 0.030 |
| colored_noise | 254 | 0.25 | 0.017 |
| harmonic_complex | 241 | 0.44 | 0.027 |
| pure_tone | 236 | 0.68 | 0.032 |
| tone_in_noise | 256 | 0.52 | 0.055 |
| white_noise | 248 | 0.18 | 0.017 |
| am_tone | 1000 | 1.10 | 0.028 |

Runtime on CPU: MOSQITO 45.8 ms/clip, MetricNet 1.31 ms/clip (batched, including front end).
