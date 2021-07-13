CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

export CUDA_VISIBLE_DEVICES="0,1"
export PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model/albert_base
export DATA_DIR=$CURRENT_DIR/data
TASK_NAME="consult"

cd $CURRENT_DIR
echo "Start running..."
python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/$TASK_NAME \
  --vocab_file=$PRETRAINED_MODELS_DIR/vocab.txt \
  --bert_config_file=$PRETRAINED_MODELS_DIR/albert_config_base.json \
  --init_checkpoint=$PRETRAINED_MODELS_DIR/albert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_epochs=100 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output/
