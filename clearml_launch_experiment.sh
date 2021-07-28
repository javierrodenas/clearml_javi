clearml-task \
--project "Test" \
--name "Test" \
--docker erenbalatkan/pytorch:1.9 \
--docker_bash_setup_script clearml_docker_setup_script.sh \
--docker_args="--shm-size=16g --ipc=host -v /media:/media" \
--script clearml_experiment_runner.py \
--queue ub_196