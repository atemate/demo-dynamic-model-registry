set -euv

echo EXPERIMENT_NAME=${EXPERIMENT_NAME}  # required

for learning_rate in 0.01 0.05 0.1 0.3 0.5; do
    echo learning_rate=$learning_rate
    for max_depth in 5 10 15; do
        echo max_depth=$max_depth
        for n_estimators in 5 10 20 50; do
            echo n_estimators=$n_estimators
            
            python model/train.py \
                --experiment_name=$EXPERIMENT_NAME \
                --run_name="lr-$learning_rate-md-$max_depth-ne-$n_estimators" \
                --learning_rate $learning_rate \
                --max_depth $max_depth \
                --n_estimators $n_estimators

        done
    done
done
