d2680423c7f8b03ce444cbad23b9ec294b503d4f -> 520
ba723d48e4d71ec9d64d57baba5a2f552ae23ce6 -> 3000
4d0d5da67e260530775f8f3cfb56a30f672d2580 -> 3700
65d703c42c5b5a4c44fa05b4538945d6cfff93f9 -> 850

같은 설정(Epoch 1로 수정) -> 모델 롤백 -> 26시간
paged_adamw_8bit 삭제 -> 23시간
optim adamw_torch_fused 으로 수정 -> 23시간
