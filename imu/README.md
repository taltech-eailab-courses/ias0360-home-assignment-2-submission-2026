# HA2 IMU gesture recognition

## Start the environment

From the repository root, start the course environment:

```bash
./build_in_docker.sh
```

In the container shell, start JupyterLab:

```bash
cd /root
jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=''
```

Open `http://localhost:8888` and navigate to
`submission/ha2/imu/ha2_imu_training.ipynb`.

## Train and compare models

Run the notebook cells from top to bottom. They:

1. Prepare the recordings and fixed train/validation/test split.
2. Train and compare the baseline CNN, separable CNN, tuned CNN, and six LSTM
   configurations.
3. Convert the models to integer TFLite and train the pruning and QAT versions
   of the best CNN-family model.
4. Evaluate the candidates and write the comparison table to
   `artifacts/experiment_results.csv`.
5. Export the selected model and its matching preprocessing settings.

The default `DEPLOY_VARIANT` is `baseline_cnn_qat`. To change training
configurations, edit `ARCHITECTURE_RUNS` in `training.py`, then restart the
notebook kernel and run the notebook cells again.

The trained models are saved in `artifacts/keras/`; quantized candidates are
saved as `artifacts/*_int8.tflite`. The comparison table is
`artifacts/experiment_results.csv`.

## Export a saved model without retraining

Set `DEPLOY_VARIANT` in the notebook's setup cell to a saved model name, for
example:

```python
DEPLOY_VARIANT = 'baseline_cnn_qat'
```

Run the setup cell, the model selector cell, and **Export the selected Pico
model**. This loads the corresponding `artifacts/<name>_int8.tflite`; it does
not retrain the model.

## Optional: run inference on the Pico

The assignment can be completed without live Pico inference. To run the live
classifier, export the model first. If building from a separate VBox checkout,
copy these generated files to the same relative paths there:

- `pico/lib/model/gesture_model_data.cpp`
- `pico/lib/model/model_settings.h`
- `pico/src/preprocessing_settings.hpp`

In the container, build the live firmware:

```bash
cd /root/submission/ha2/imu/pico
cmake -S . -B build -DPICO_BOARD=pico_w -DPICO_SDK_PATH=/opt/pico-sdk
cmake --build build --target main -j4
```

Flash `pico/build/main.uf2` to the Pico W. Open its USB serial port at
115200 baud. When it prints `Perform a gesture now`, perform a gesture during
the following three-second recording window. The serial output includes the
predicted class and inference time.

To record Pico latency, save several serial outputs to
`artifacts/pico_serial.log`, then run the notebook's **Record Pico inference
latency** cell for the selected model.

## Optional: compare longer input lengths on the host

From the container:

```bash
cd /root/submission/ha2/imu
python run_length_sweep.py
```

The sweep writes its models and comparison results under `artifacts_len256/`
and `artifacts_len512/`. These longer-input models are for host comparison;
the notebook's Pico export uses the 128-step models.
