"""Home Assignment 2: controlled MNIST experiments, accuracy objective.

All data caches, models and reports stay beneath this script's directory.
Training is explicit (--preset); --smoke uses synthetic data only. Own-digit
accuracy/timing remain pending until ten actual touchscreen samples are supplied.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('KERAS_HOME',str(ROOT/'.cache/keras'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')

import numpy as np
import tensorflow as tf
import tf_keras as keras

from digits import load_own_digits,save_preview
from models import build_model


def experiment_plan(preset):
    baseline=dict(architecture='custom_cnn',epochs=5,batch=64,learning_rate=.001,
                  optimizer='adam',filters=[8,16],dense=32,dropout=0.0)
    runs=[dict(baseline,name='baseline')]
    if preset=='accuracy':
        for name,changes in [
            ('more_epochs',{'epochs':10}),
            ('lower_learning_rate',{'learning_rate':.0003}),
            ('dropout',{'dropout':.25}),
            ('wider_cnn',{'filters':[16,32],'dense':64}),
            ('sgd_momentum',{'optimizer':'sgd','learning_rate':.01})]:
            runs.append(dict(baseline,name=name,**changes))
    runs.extend([dict(baseline,name='lenet',architecture='lenet'),
                 dict(baseline,name='mlp',architecture='mlp')])
    if preset=='quick':
        for run in runs:run['epochs']=2
    return runs


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def setup(seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)


def optimizer_for(spec):
    if spec['optimizer']=='adam':return keras.optimizers.Adam(spec['learning_rate'])
    return keras.optimizers.SGD(spec['learning_rate'],momentum=.9)


def compile_model(model,spec):
    model.compile(optimizer=optimizer_for(spec),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


def dataset(images,labels,batch,shuffle=False):
    data=tf.data.Dataset.from_tensor_slices((images,labels))
    if shuffle:data=data.shuffle(len(images),seed=42,reshuffle_each_iteration=True)
    options=tf.data.Options()
    options.threading.private_threadpool_size=2
    options.threading.max_intra_op_parallelism=1
    return data.batch(batch).with_options(options).prefetch(1)


def load_mnist():
    # The official test set is reserved, not reused for hyperparameter selection.
    (x,y),(test_x,test_y)=keras.datasets.mnist.load_data()
    indices=np.random.default_rng(42).permutation(len(x))
    val_ids,train_ids=indices[:6000],indices[6000:]
    train_x=x[train_ids].astype('float32')[...,None]/255
    val_x=x[val_ids].astype('float32')[...,None]/255
    test_x=test_x.astype('float32')[...,None]/255
    return (train_x,y[train_ids]),(val_x,y[val_ids]),(test_x,test_y),train_ids,val_ids


def prediction_timing(model,images):
    """One complete ten-image serial pass; report total seconds divided by ten."""
    assert len(images)==10
    @tf.function(input_signature=[tf.TensorSpec((1,28,28,1),tf.float32)])
    def infer(batch):return model(batch,training=False)
    for _ in range(3):infer(tf.convert_to_tensor(images[:1])).numpy()
    predictions=[]
    start=time.perf_counter()
    for image in images:
        probabilities=infer(tf.convert_to_tensor(image[None])).numpy()
        predictions.append(int(probabilities[0].argmax()))
    elapsed=time.perf_counter()-start
    return {'total_prediction_seconds':elapsed,'average_prediction_ms':elapsed*1000/10,
            'predictions':predictions,'method':'Three warmups, then 10 serial batch-1 calls; '
                     'total timed prediction seconds / 10. Excludes disk I/O and preprocessing.'}


def render_results(output,rows,own_status):
    with (output/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    lines=['# Home Assignment 2 experiment results','',
           'Optimization objective: validation accuracy. Each selected checkpoint maximizes validation accuracy within its run.',
           '', '| Experiment | Architecture | Epochs | Own-digit avg ms | Parameters / weight KiB / model H5 KiB | Validation accuracy | Own-digit accuracy |',
           '|---|---|---:|---:|---|---:|---:|']
    for r in rows:
        own_ms='pending' if r['own_average_ms'] is None else f"{r['own_average_ms']:.3f}"
        own_acc='pending' if r['own_accuracy'] is None else f"{r['own_accuracy']:.0%}"
        memory=f"{r['parameters']:,} / {r['weight_bytes']/1024:.1f} / {r['model_file_bytes']/1024:.1f}"
        lines.append(f"| {r['experiment']} | {r['architecture']} | {r['epochs']} | {own_ms} | {memory} | {r['validation_accuracy']:.2%} | {own_acc} |")
    lines.extend(['','Own-data status: '+own_status['status']+'. Missing values are not zero accuracies.',
                  'Weight bytes and H5 bytes are separate measures: the serialized model already contains weights, so do not add them.',
                  'Model summaries are stored in each experiment directory. This does not measure peak runtime RAM.',
                  'The separate MNIST-reference timing column in results.csv uses ten validation examples, not personal data.',
                  'Own-data timing uses total warmed prediction time for ten digits divided by ten, excluding preprocessing.',
                  'With only one example per digit, own-data accuracy changes in 10 percentage-point steps.'])
    if len(rows)>1:
        winner=max(rows,key=lambda r:r['validation_accuracy'])
        baseline=next(r for r in rows if r['experiment']=='baseline')
        lines.extend(['',f"Best validation result: {winner['experiment']} ({winner['validation_accuracy']:.2%}).",
                      f"Observed improvement over baseline: {(winner['validation_accuracy']-baseline['validation_accuracy'])*100:.3f} percentage points.",
                      'The final summary identifies the best candidate within each architecture before testing the reserved MNIST test set.'])
    summary_path=output/'final_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        lines.extend(['','## Architecture comparison on the reserved test set','',
                      '| Architecture | Validation-selected experiment | Test accuracy |',
                      '|---|---|---:|'])
        for family,value in summary['architecture_test_results'].items():
            lines.append(f"| {family} | {value['experiment']} | {value['test_accuracy']:.2%} |")
        lines.extend(['','These are results for one fixed split and seed. The custom CNN had more hyperparameter trials than the alternatives; this is not a matched-search-budget proof of architectural superiority.',
                      'The real touchscreen/SD-card evaluation remains required when own-data status is pending.'])
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def smoke(output):
    results={}
    rng=np.random.default_rng(42)
    x=rng.random((4,28,28,1),dtype=np.float32);y=np.arange(4)
    for name in ('custom_cnn','lenet','mlp'):
        setup(42);model=build_model(name)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        loss,accuracy=model.train_on_batch(x,y)
        predictions=model(x,training=False).numpy()
        assert predictions.shape==(4,10) and np.isfinite(predictions).all() and np.isfinite(loss)
        np.testing.assert_allclose(predictions.sum(axis=1),1,atol=1e-5)
        results[name]={'parameters':model.count_params(),'output_shape':list(predictions.shape),'finite_loss':True}
    save_json(output/'smoke_check.json',{'status':'passed','scope':'Synthetic software check only; no MNIST accuracy claim.',
                                         'tensorflow':tf.__version__,'models':results,
                                         'runtime':'docker' if Path('/.dockerenv').exists() else 'host'})
    print(json.dumps(results,indent=2))


def run(args,output):
    (train_x,train_y),(val_x,val_y),(test_x,test_y),train_ids,val_ids=load_mnist()
    np.savez_compressed(output/'split_indices.npz',train=train_ids,val=val_ids)
    own,own_status=load_own_digits(ROOT/'data/own_digits')
    save_json(output/'own_data_status.json',own_status)
    if own is not None:save_preview(own,output/'own_digits')
    reference=np.stack([val_x[np.flatnonzero(val_y==digit)[0]] for digit in range(10)])
    rows=[]
    for spec in experiment_plan(args.preset):
        print('\nEXPERIMENT '+spec['name'],flush=True)
        setup(42);directory=output/spec['name'];directory.mkdir()
        save_json(directory/'settings.json',spec)
        model=build_model(spec['architecture'],spec['filters'],spec['dense'],spec['dropout'])
        compile_model(model,spec)
        summary=[];model.summary(print_fn=summary.append)
        (directory/'model_summary.txt').write_text('\n'.join(summary)+'\n')
        best=directory/'best.weights.h5'
        checkpoint=keras.callbacks.ModelCheckpoint(str(best),monitor='val_accuracy',mode='max',
                                                   save_best_only=True,save_weights_only=True)
        start=time.perf_counter()
        history=model.fit(dataset(train_x,train_y,spec['batch'],True),
                          validation_data=dataset(val_x,val_y,128),epochs=spec['epochs'],callbacks=[checkpoint],verbose=2)
        elapsed=time.perf_counter()-start
        model.load_weights(best)
        val_accuracy=float(model.evaluate(dataset(val_x,val_y,128),verbose=0)[1])
        saved=directory/'model.h5';model.save(str(saved),include_optimizer=False)
        save_json(directory/'history.json',history.history)
        reference_time=prediction_timing(model,reference)
        own_result=prediction_timing(model,own) if own is not None else None
        if own_result:
            own_result['accuracy']=float(np.mean(np.array(own_result['predictions'])==np.arange(10)))
            save_json(directory/'own_predictions.json',own_result)
        row={'experiment':spec['name'],'architecture':spec['architecture'],'epochs':spec['epochs'],
             'batch':spec['batch'],'learning_rate':spec['learning_rate'],'optimizer':spec['optimizer'],
             'dropout':spec['dropout'],'parameters':int(model.count_params()),
             'weight_bytes':sum(int(np.prod(w.shape))*w.dtype.size for w in model.weights),
             'model_file_bytes':saved.stat().st_size,'training_seconds':elapsed,
             'validation_accuracy':val_accuracy,'own_average_ms':own_result['average_prediction_ms'] if own_result else None,
             'own_accuracy':own_result['accuracy'] if own_result else None,
             'own_total_seconds':own_result['total_prediction_seconds'] if own_result else None,
             'mnist_reference_average_ms':reference_time['average_prediction_ms'],
             'model_sha256':hashlib.sha256(saved.read_bytes()).hexdigest()}
        rows.append(row);render_results(output,rows,own_status)
        save_json(output/'results.json',rows)
        print('Saved '+str(saved),flush=True)
    # Compare each architecture's validation-selected candidate on the reserved
    # test set only after all hyperparameter runs are finished.
    family_tests={}
    for family in ('custom_cnn','lenet','mlp'):
        best=max([r for r in rows if r['architecture']==family],key=lambda r:r['validation_accuracy'])
        model=keras.models.load_model(str(output/best['experiment']/'model.h5'),compile=False)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        family_tests[family]={'experiment':best['experiment'],
                              'test_accuracy':float(model.evaluate(dataset(test_x,test_y,128),verbose=0)[1])}
    winner=max(rows,key=lambda r:r['validation_accuracy'])
    model=keras.models.load_model(str(output/winner['experiment']/'model.h5'),compile=False)
    model.save(str(output/'best_saved_model'),include_optimizer=False)
    save_json(output/'final_summary.json',{'status':'complete' if own is not None else 'training_complete_own_data_pending','objective':'accuracy',
              'preset':args.preset,'best_validation_experiment':winner['experiment'],
              'architecture_test_results':family_tests,'own_data':own_status,
              'samples':{'train':54000,'validation':6000,'test':10000},
              'seed':42,'tensorflow':tf.__version__,'tf_keras':keras.__version__,
              'python':platform.python_version(),'machine':platform.machine(),
              'runtime_platform':platform.platform(),'cpu_threads':4,
              'runtime':'docker' if Path('/.dockerenv').exists() else 'host',
              'note':'Own-data metrics pending unless ten real collected digits were present. '
                     'Quick preset checks architecture comparison only, not the optimization sweep.'})
    render_results(output,rows,own_status)
    print(json.dumps(family_tests,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=['quick','accuracy'])
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.plan:
        print(json.dumps(experiment_plan(args.preset or 'accuracy'),indent=2));return
    if not args.smoke and not args.preset:
        parser.print_help();return
    output=args.output or Path('outputs')/time.strftime('run_%Y%m%d_%H%M%S')
    output=(ROOT/output).resolve()
    if ROOT/'outputs' not in output.parents:raise ValueError('Outputs must stay under this project outputs/ directory')
    output.mkdir(parents=True,exist_ok=False)
    if args.smoke:smoke(output)
    else:run(args,output)
    print('Outputs: '+str(output),flush=True)


if __name__=='__main__':main()
