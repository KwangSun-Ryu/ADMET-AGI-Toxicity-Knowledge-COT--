export INFER_BASE_URL=http://192.168.0.202:25321
export INFER_PATH=/generate


export # 요청 기본값(없으면 코드 default 사용)
export MAX_TOKENS=4096
export TEMP=0.0
export TOP_P=1.0

export RETRIES=3
export LOG_DIR=./logs

export INPUT_JSON=./test.json
export OUTPUT_JSON=./test_model_res.json

python main.py \
  --input $INPUT_JSON \
  --output $OUTPUT_JSON \
  --log_dir  $LOG_DIR \
  --batch_size 32
