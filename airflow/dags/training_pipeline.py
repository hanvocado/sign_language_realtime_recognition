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
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

from airflow.dags.training.config import DEFAULT_ARGS
from airflow.dags.training.sensors import IcebergGoldVersionSensor
from airflow.dags.training.tasks.context    import prepare_run_context
from airflow.dags.training.tasks.retrain    import retrain_check, branch_on_decision
from airflow.dags.training.tasks.data       import download_gold_snapshot, split_dataset
from airflow.dags.training.tasks.training   import train_model
from airflow.dags.training.tasks.evaluation import evaluate_model
from airflow.dags.training.tasks.registry   import promote_or_skip
from airflow.dags.preprocessing_pipeline    import task_failure_email_alert

with DAG(
    dag_id="training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Auto-train SLR model when a new Gold snapshot is published",
    schedule_interval=timedelta(hours=1),
    catchup=False,
    on_failure_callback=task_failure_email_alert,
    tags=["training", "sign-language", "mlflow"],
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
