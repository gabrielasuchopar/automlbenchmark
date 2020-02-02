import logging
import os
import numpy as np
import pickle
import pprint
import random
import sys
import tempfile as tmp

from genens.render.graph import create_graph
from genens.render.plot import export_plot
from genens.workflow.evaluate import SampleCrossValEvaluator

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from genens import GenensClassifier, GenensRegressor

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, make_scorer

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import Encoder, impute
from amlb.results import save_predictions_to_file
from amlb.utils import Timer, touch

from typing import Union


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** genens ****\n")

    is_classification = config.type == 'classification'

    if not is_classification:
        Warning("Regression not supported.")
        return None

    # Mapping of benchmark metrics to TPOT metrics
    metrics_mapping = dict(
        acc=make_scorer(accuracy_score),
        auc=make_scorer(roc_auc_score),
        f1=make_scorer(f1_score),
        logloss=make_scorer(log_loss, greater_is_better=False),
        mae=make_scorer(mean_absolute_error, greater_is_better=False),
        mse=make_scorer(mean_squared_error, greater_is_better=False),
        msle=make_scorer(mean_squared_log_error, greater_is_better=False),
        r2=make_scorer(r2_score)
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train, X_test = impute(dataset.train.X_enc, dataset.test.X_enc)
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    sample_size = config.framework_params.get('_sample_size', None)
    if sample_size is not None:
        evaluator = SampleCrossValEvaluator(sample_size=sample_size, per_gen=True, cv_k=5)
        training_params['evaluator'] = evaluator

    runtime_s = config.max_runtime_seconds
    runtime_s -= 5 * 60  # avoid premature process termination
    print(f"Setting time limit to {runtime_s} minutes.")

    log.info('Running genens with a maximum time of %ss on %s cores, optimizing %s.',
             runtime_s, n_jobs, scoring_metric)

    if config.seed is not None:
        # random state is yet to be unified in genens
        np.random.seed(config.seed)
        random.seed(config.seed)

    print(f'Training params: {training_params}')

    estimator = GenensClassifier if is_classification else GenensRegressor
    genens_est = estimator(n_jobs=n_jobs,
                           max_evo_seconds=runtime_s,
                           scorer=scoring_metric,
                           # random_state=config.seed,
                           **training_params)

    with Timer() as training:
        genens_est.fit(X_train, y_train)

    log.info('Predicting on the test set.')
    predictions = genens_est.predict(X_test)

    try:
        best_pipe = genens_est.get_best_pipelines()[0]
        probabilities = best_pipe.predict_proba(X_test) if is_classification else None
    except AttributeError:
        target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
        probabilities = Encoder('one-hot', target=False, encoded_type=float).fit(target_values_enc).transform(predictions)

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=is_classification)

    save_artifacts(genens_est, config)

    # TODO
    return dict(
        models_count=len(estimator.get_best_pipelines()),
        training_duration=training.duration
    )


def make_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


def save_artifacts(estimator: Union[GenensClassifier, GenensRegressor], config):
    try:
        artifacts = config.framework_params.get('_save_artifacts', False)

        if 'models' in artifacts:
            models_dir = os.path.join(make_subdir('models', config))

            # pickle top 3 best pipelines
            for i, pipe in enumerate(estimator.get_best_pipelines()):
                with open(models_dir + '/pipeline{}.pickle'.format(i), 'wb') as pickle_file:
                    pickle.dump(pipe, pickle_file, pickle.HIGHEST_PROTOCOL)

            # top 3 individual fitness values
            with open(models_dir + '/ind-fitness.txt', 'w+') as out_file:
                best_inds = estimator.get_best_pipelines(as_individuals=True)

                for i, ind in enumerate(best_inds):
                    out_file.write('Individual {}: Score {}\n'.format(i, ind.fitness.values))
                    # individual tree
                    create_graph(ind, models_dir + '/graph{}.png'.format(i))

        if 'log' in artifacts:
            log_dir = os.path.join(make_subdir('logs', config))

            # write logbook string representation to output dir
            with open(log_dir + '/logbook.txt', 'w+') as log_file:
                log_file.write(estimator.logbook.__str__() + '\n')

            # evolution plot
            export_plot(estimator, log_dir + '/result.png')

    except:
        log.debug("Error when saving artifacts.", exc_info=True)
