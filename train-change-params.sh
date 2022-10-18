set -euv

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
echo TIMESTAMP=$TIMESTAMP

for learning_rate in 0.01 0.05 0.1 0.3 0.5; do
    echo learning_rate=$learning_rate
    for max_depth in 5 10 15; do
        echo max_depth=$max_depth
        for n_estimators in 5 10 20 50; do
            echo n_estimators=$n_estimators
            
            python train.py \
                --experiment_name="train-change-params-$TIMESTAMP" \
                --run_name="lr-$learning_rate-md-$max_depth-ne-$n_estimators" \
                --learning_rate $learning_rate \
                --max_depth $max_depth \
                --n_estimators $n_estimators \
                --timestamp $TIMESTAMP

        done
    done
done
