
experiment_name="train-$(date +"%Y%m%d%H%M%S")"

for learning_rate in 0.01 0.05 0.1 0.3 0.5; do
    echo learning_rate=$learning_rate
    for max_depth in 5 10 15; do
        echo max_depth=$max_depth
        for n_estimators in 5 10 20 50; do
            echo n_estimators=$n_estimators
            
            gh workflow run train-model \
                --ref=$(git branch --show-current) \
                -f experiment_name=$experiment_name \
                -f learning_rate=$learning_rate \
                -f max_depth=$max_depth \
                -f n_estimators=$n_estimators

        done
    done
done

