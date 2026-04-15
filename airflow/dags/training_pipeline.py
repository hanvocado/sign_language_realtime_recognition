"""
Sign Language Recognition — Training Pipeline DAG

Triggered automatically when preprocessing_pipeline publishes a new Gold
snapshot version to the Iceberg inventory table.

Task flow:
    start
      └─ iceberg_gold_sensor           poll Iceberg for new gold_version
           └─ training_prepare_run_context
                └─ training_retrain_check    skip / finetune / full
                     └─ branch_on_decision
                          ├─ skip_training          (EmptyOperator)
                          └─ download_gold_snapshot
                               └─ split_dataset
                                    └─ train_model
                                         └─ evaluate_model
                                              └─ promote_or_skip
                                                   └─ end

All task logic lives in airflow/dags/training/:
    config.py       environment constants + DEFAULT_ARGS
    utils.py        subprocess, Spark, run context, gold state helpers
    sensors.py      IcebergGoldVersionSensor
    tasks/
        context.py      prepare_run_context
        retrain.py      retrain_check, branch_on_decision
        data.py         download_gold_snapshot, split_dataset
        training.py     train_model
        evaluation.py   evaluate_model
        registry.py     promote_or_skip
"""

from datetime import timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

from shared.config import DEFAULT_ARGS
from shared.alerts import task_failure_email_alert
from training.sensors import IcebergGoldVersionSensor
from training.tasks.context import prepare_run_context
from training.tasks.retrain import retrain_check, branch_on_decision
from training.tasks.data import download_gold_snapshot, split_dataset
from training.tasks.training import train_model
from training.tasks.evaluation import evaluate_model
from training.tasks.registry import promote_or_skip
from training.config import (
    MODEL_TYPE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    SEQ_LEN,
    BATCH_SIZE,
    LR,
    FINETUNE_LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    EPOCHS,
    PATIENCE,
    NUM_WORKERS,
)

with DAG(
    dag_id="training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Manual SLR training pipeline (dev mode)",
    schedule_interval=None,
    catchup=False,
    on_failure_callback=task_failure_email_alert,
    tags=["training", "sign-language", "mlflow"],
    params={
        "model_type": Param(
            MODEL_TYPE,
            type="string",
            enum=["lstm", "gru", "transformer"],
            title="Model type",
            description="Architecture used by src/pipeline/train_mlflow.py.",
        ),
        "hidden_dim": Param(
            HIDDEN_DIM,
            type="integer",
            minimum=1,
            title="Hidden dim",
        ),
        "num_layers": Param(
            NUM_LAYERS,
            type="integer",
            minimum=1,
            title="Num layers",
        ),
        "dropout": Param(
            DROPOUT,
            type="number",
            minimum=0.0,
            maximum=1.0,
            title="Dropout",
        ),
        "seq_len": Param(
            SEQ_LEN,
            type="integer",
            minimum=1,
            title="Sequence length",
        ),
        "batch_size": Param(
            BATCH_SIZE,
            type="integer",
            minimum=1,
            title="Batch size",
        ),
        "lr": Param(
            LR,
            type="number",
            minimum=0.0,
            title="Learning rate",
        ),
        "finetune_lr": Param(
            FINETUNE_LR,
            type="number",
            minimum=0.0,
            title="Fine-tune learning rate",
            description="Used when retrain decision is 'finetune'.",
        ),
        "weight_decay": Param(
            WEIGHT_DECAY,
            type="number",
            minimum=0.0,
            title="Weight decay",
        ),
        "label_smoothing": Param(
            LABEL_SMOOTHING,
            type="number",
            minimum=0.0,
            maximum=1.0,
            title="Label smoothing",
        ),
        "epochs": Param(
            EPOCHS,
            type="integer",
            minimum=1,
            title="Epochs",
        ),
        "patience": Param(
            PATIENCE,
            type="integer",
            minimum=0,
            title="Early-stopping patience",
        ),
        "num_workers": Param(
            NUM_WORKERS,
            type="integer",
            minimum=0,
            title="DataLoader num_workers",
        ),
    },
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    iceberg_sensor = IcebergGoldVersionSensor(
        task_id="iceberg_gold_sensor",
        poke_interval=300,
        timeout=3600 * 4,
        mode="reschedule",
        soft_fail=False,
    )

    prepare_context = PythonOperator(
        task_id="training_prepare_run_context",
        python_callable=prepare_run_context,
    )

    check = PythonOperator(
        task_id="training_retrain_check",
        python_callable=retrain_check,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_decision",
        python_callable=branch_on_decision,
    )

    skip = EmptyOperator(task_id="skip_training")

    with TaskGroup(group_id="training", prefix_group_id=False):
        download = PythonOperator(
            task_id="download_gold_snapshot",
            python_callable=download_gold_snapshot,
        )
        split = PythonOperator(
            task_id="split_dataset",
            python_callable=split_dataset,
        )
        train = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            execution_timeout=timedelta(hours=6),
        )
        evaluate = PythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate_model,
        )
        promote = PythonOperator(
            task_id="promote_or_skip",
            python_callable=promote_or_skip,
        )

        download >> split >> train >> evaluate >> promote

    (
        start
        >> iceberg_sensor
        >> prepare_context
        >> check
        >> branch
        >> [skip, download]
    )
    skip    >> end
    promote >> end
