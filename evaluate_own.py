"""Fill in real-own-digit accuracy and timing after SD-card images are added."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_experiments import ROOT,keras,load_own_digits,prediction_timing,render_results,save_json,save_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,help='Existing outputs/run_... or results/study directory')
    args=parser.parse_args()
    output=(ROOT/args.run).resolve()
    if not any(allowed in output.parents for allowed in (ROOT/'outputs',ROOT/'results')):
        raise ValueError('Use a run inside this project outputs/ or results/ directory')
    own,status=load_own_digits(ROOT/'data/own_digits')
    if own is None:raise ValueError('Missing real digit samples: '+str(status['missing_digits']))
    rows=json.loads((output/'results.json').read_text())
    save_preview(own,output/'own_digits')
    for row in rows:
        model=keras.models.load_model(str(output/row['experiment']/'model.h5'),compile=False)
        result=prediction_timing(model,own)
        result['accuracy']=float(np.mean(np.array(result['predictions'])==np.arange(10)))
        save_json(output/row['experiment']/'own_predictions.json',result)
        row.update(own_average_ms=result['average_prediction_ms'],own_accuracy=result['accuracy'],
                   own_total_seconds=result['total_prediction_seconds'])
    save_json(output/'results.json',rows);save_json(output/'own_data_status.json',status)
    summary_path=output/'final_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());summary['own_data']=status;save_json(summary_path,summary)
    render_results(output,rows,status)
    print('Updated own-data results. Inspect outputs/your_run/own_digits/preview_0_to_9.png.')


if __name__=='__main__':main()
