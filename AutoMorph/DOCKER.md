## Running with Docker

First, create a folder named images and put all images inside. Prepare the resolution csv for all images, an example can be seen at resolution_information.csv. More information can be found [here](https://github.com/rmaphoh/AutoMorph?tab=readme-ov-file#pixel-resolution:~:text=removed%20unused%20files.-,Pixel%20resolution,-The%20units%20for) 
```
├──images
    ├──1.jpg
    ├──2.jpg
    ├──3.jpg
├──resolution_information.csv
``` 

Then, pull the [docker image](https://hub.docker.com/repository/docker/yukunzhou/image_automorph/general) and run the tool.
```bash
docker pull yukunzhou/image_automorph:latest
```

Please substitute the `{images_path}` with the path to `images` folder, e.g. `/home/AutoMorph/images`. And replace the `{results_path}` with the path that you want to save the results, e.g. `/home/AutoMorph/Results`.

```bash
docker run --rm   --shm-size=2g   -v {images_path}:/app/AutoMorph/images   -v {results_path}:/app/AutoMorph/Results   -ti   --runtime=nvidia   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility   -e NVIDIA_VISIBLE_DEVICES=all   yukunzhou/image_automorph
```



