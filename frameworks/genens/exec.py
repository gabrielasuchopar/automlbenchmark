import logging
import os
import numpy as np
import pickle
import random
import sys
import tempfile as tmp

from genens.render.graph import create_graph
from genens.render.plot import export_plot
from genens.workflow.evaluate import SampleCrossValEvaluator, CrossValEvaluator

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from genens import GenensClassifier, GenensRegressor

from sklearn.metrics import get_scorer

from frameworks.shared.callee import call_run, result, output_subdir, utils

from typing import Union


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** genens ****\n")

    is_classification = config.type == 'classification'

    if not is_classification:
        Warning("Regression not supported.")
        return None

    # Mapping of benchmark metrics to TPOT metrics
    metrics_mapping = {
        'acc': get_scorer('accuracy'),
        'auc': get_scorer('roc_auc'),
        'f1': get_scorer('f1'),
        'logloss': get_scorer('neg_log_loss'),
        'mae': get_scorer('neg_mean_absolute_error'),
        'mse': get_scorer('neg_mean_squared_error'),
        'msle': get_scorer('neg_mean_squared_log_error'),
        'r2': get_scorer('r2')
    }
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    sample_size = config.framework_params.get('_sample_size', None)
    if sample_size is not None:
        evaluator = SampleCrossValEvaluator(sample_size=sample_size, per_gen=True, cv_k=5)
    else:
        evaluator = CrossValEvaluator(cv_k=5)

    print(f"Chosen sample size: {sample_size}.")
    print(f'cv_k: {evaluator.cv_k}')

    training_params['evaluator'] = evaluator

    runtime_s = config.max_runtime_seconds
    if runtime_s >= 600:
        runtime_s -= 5 * 60  # avoid premature process termination
    elif runtime_s > 10:
        runtime_s -= 5

    if not config.framework_params.get('disable_logging', True):
        log_path = os.path.join(output_subdir('logs', config), 'evo_log_file.txt')
    else:
        log_path = None

    print(f"Setting time limit to {runtime_s} seconds.")

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
                           log_path=log_path,
                           **training_params)

    with utils.Timer() as training:
        genens_est.fit(X_train, y_train)

    log.info('Predicting on the test set.')

    best_pipe = genens_est.get_best_pipelines()[0]
    best_pipe.fit(X_train, y_train)

    predictions = best_pipe.predict(X_test)

    try:
        probabilities = best_pipe.predict_proba(X_test) if is_classification else None
    except AttributeError:
        target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
        probabilities = utils.Encoder('one-hot', target=False, encoded_type=float).fit(target_values_enc).transform(predictions)

    save_artifacts(genens_est, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(genens_est.get_best_pipelines()),
                  training_duration=training.duration)


def save_artifacts(estimator: Union[GenensClassifier, GenensRegressor], config):
    try:
        artifacts = config.framework_params.get('_save_artifacts', False)

        if 'pickle_models' in artifacts:
            models_dir = os.path.join(output_subdir('pickle_models', config))

            # pickle top 3 best pipelines
            for i, pipe in enumerate(estimator.get_best_pipelines()):
                with open(models_dir + '/pipeline{}.pickle'.format(i), 'wb') as pickle_file:
                    pickle.dump(pipe, pickle_file, pickle.HIGHEST_PROTOCOL)

        if 'models' in artifacts:
            models_dir = os.path.join(output_subdir('models', config))

            # top 3 individual fitness values
            with open(models_dir + '/ind-fitness.txt', 'w+') as out_file:
                best_inds = estimator.get_best_pipelines(as_individuals=True)

                for i, ind in enumerate(best_inds):
                    out_file.write('Individual {}: Score {}\n'.format(i, ind.fitness.values))
                    # individual tree
                    create_graph(ind, models_dir + '/graph{}.png'.format(i))

        if 'log' in artifacts:
            log_dir = os.path.join(output_subdir('logs', config))

            # write logbook string representation to output dir
            with open(log_dir + '/logbook.txt', 'w+') as log_file:
                log_file.write(estimator.logbook.__str__() + '\n')

            # evolution plot
            export_plot(estimator, log_dir + '/result.png')

    except:
        log.debug("Error when saving artifacts.", exc_info=True)

if __name__ == '__main__':
    call_run(run)