```sh
mkdir build
cd build
cmake .. -DDIGIT_LABEL=1
make -j$(nproc)
```

```sh
cd ~
chmod +x flash.sh   # only needed once per container
./flash.sh blink_example/build/blink.uf2
```