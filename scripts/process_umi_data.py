import os
import sys
import subprocess
import time
import multiprocessing as mp
from typing import cast
import zarr

PROJECT_NAME = "uva"

DATASETS = {
    ### UMI
    "dish_washing_0": "https://real.stanford.edu/umi/data/dish_washing/bimanual_dish_washing.zarr.zip",
    "cloth_folding_0": "https://real.stanford.edu/umi/data/bimanual_cloth_folding/bimanual_cloth_folding.zarr.zip",
    "dynamic_tossing_0": "https://real.stanford.edu/umi/data/dynamic_tossing/dynamic_tossing.zarr.zip",
    "cup_arrangement_0": "https://real.stanford.edu/umi/data/cup_in_the_wild/cup_in_the_wild.zarr.zip",
    "cup_arrangement_1": "https://real.stanford.edu/umi/data/cup_arrangement/cup_in_the_lab.zarr.zip",

    ### ManiWAV
    "whiteboard_wiping_0": "https://real.stanford.edu/maniwav/data/wipe/replay_buffer.zarr.zip",
    "bagle_flipping_0": "https://real.stanford.edu/maniwav/data/flip/replay_buffer.zarr.zip",
    "bagle_flipping_1": "https://real.stanford.edu/maniwav/data/bagel_in_wild/replay_buffer.zarr.zip",
    "dice_pouring_0": "https://real.stanford.edu/maniwav/data/pour/replay_buffer.zarr.zip",
    "wire_strapping_0": "https://real.stanford.edu/maniwav/data/velcro_tape/replay_buffer.zarr.zip",

    ### UMI-on-Legs
    "kettlebell_pushing_0": "https://real.stanford.edu/umi-on-legs/pushing_2024_05_29_huy.zarr.zip",
    "tennis_ball_tossing_0": "https://real.stanford.edu/umi-on-legs/tossing.zarr.zip",

    "charger_unplugging_0": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/unplug_charger/dataset.zarr.zip?download=true",
    "water_pouring_0": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/pour_water/dataset.zarr.zip?download=true",
    "water_pouring_1": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/pour_water_16_env_4_object/dataset_part_aa?download=true;https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/pour_water_16_env_4_object/dataset_part_ab?download=true", # Merge the two parts before unzipping
    "mouse_arrangement_0": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/arrange_mouse/dataset.zarr.zip?download=true",
    "mouse_arrangement_1": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/arrange_mouse_16_env_4_object/dataset.zarr.zip?download=true",
    "towel_folding_0": "https://huggingface.co/datasets/Fanqi-Lin/Processed-Task-Dataset/resolve/main/fold_towel/dataset.zarr.zip?download=true",
}

def download_data(dataset_name: str, url: str, output_dir: str) -> None:
    """
    Download the data from the given URL and save it to the given dataset name.
    """
    os.makedirs(output_dir, exist_ok=True)
    shm_data_dir = f"/dev/shm/{PROJECT_NAME}/temp"
    if ";" in url:
        urls = url.split(";")
        os.makedirs(shm_data_dir, exist_ok=True)
        def download_url(url: str, id: int, output_dir: str) -> None:
            if os.path.exists(f"{output_dir}/{dataset_name}_part_{id}"):
                print(f"Skipping downloading {dataset_name} because {output_dir}/{dataset_name}_part_{id} already exists")
            else:
                print(f"Downloading {dataset_name} from {url} to {output_dir}/{dataset_name}_part_{id}")
                subprocess.run(["wget", url, "-O", f"{output_dir}/{dataset_name}_part_{id}"], check=True, stdout=subprocess.DEVNULL)
            print(f"Moving {output_dir}/{dataset_name}_part_{id} to {shm_data_dir}/{dataset_name}_part_{id}")
            subprocess.run(["mv", f"{output_dir}/{dataset_name}_part_{id}", shm_data_dir], check=True)

        # with mp.Pool(len(urls)) as pool:
        #     pool.starmap(download_url, [(url, i, output_dir) for i, url in enumerate(urls)])
        for i, url in enumerate(urls):
            download_url(url, i, output_dir)

        print(f"Merging {dataset_name}.zarr.zip")
        subprocess.run(["cat", f"{shm_data_dir}/{dataset_name}_part_*", ">", f"{shm_data_dir}/{dataset_name}.zarr.zip"], shell=True, check=True, stdout=subprocess.DEVNULL)
        print(f"Moving {shm_data_dir}/{dataset_name}.zarr.zip to {output_dir}/{dataset_name}.zarr.zip")
        subprocess.run(["mv", f"{shm_data_dir}/{dataset_name}.zarr.zip", output_dir], check=True)
        subprocess.run(["rm", "-rf", shm_data_dir], check=True)

    else:
        print(f"Downloading {dataset_name} from {url} to {output_dir}/{dataset_name}.zarr.zip")
        subprocess.run(["wget", url, "-O", f"{output_dir}/{dataset_name}.zarr.zip"], check=True, stdout=subprocess.DEVNULL)
        print(f"Downloaded {dataset_name} to {output_dir}/{dataset_name}.zarr.zip")

def convert_zip_to_lz4(dataset_name: str, data_dir: str):
    # First copy zip file to shared memory /dev/shm
    shm_data_dir = f"/dev/shm/{PROJECT_NAME}/temp"
    os.makedirs(shm_data_dir, exist_ok=True)
    shm_file = f"{shm_data_dir}/{dataset_name}.zarr.zip"
    zip_file = f"{data_dir}/{dataset_name}.zarr.zip"

    print(f"Copying {zip_file} to {shm_file}")
    subprocess.run(["cp", zip_file, shm_file], check=True)

    print(f"Unzipping {shm_file} to {shm_data_dir}/{dataset_name}.zarr")
    subprocess.run(["unzip", shm_file, "-d", f"{shm_data_dir}/{dataset_name}.zarr"], check=True, stdout=subprocess.DEVNULL)
    print(f"Removing {shm_file}")
    subprocess.run(["rm", shm_file], check=True)

    print(f"Compressing {dataset_name}.zarr to {dataset_name}.zarr.tar.lz4")
    subprocess.run([f"tar cf - {dataset_name}.zarr | lz4 -c > {dataset_name}.zarr.tar.lz4"], cwd=shm_data_dir, shell=True, check=True)

    zip_file_dir = os.path.dirname(zip_file)
    if zip_file_dir.endswith("zip"):
        # data_dir/zip
        # data_dir/lz4
        zip_file_dir = os.path.dirname(zip_file_dir)
    os.makedirs(f"{zip_file_dir}/lz4", exist_ok=True)
    print(f"Copying {shm_data_dir}/{dataset_name}.zarr.tar.lz4 to {zip_file_dir}/lz4")
    subprocess.run(["cp", f"{shm_data_dir}/{dataset_name}.zarr.tar.lz4", f"{zip_file_dir}/lz4"], check=True)

    print(f"Removing {shm_data_dir}/{dataset_name}.zarr")
    subprocess.run(["rm", "-rf", f"{shm_data_dir}/{dataset_name}.zarr"], check=True)

    print(f"Removing {shm_data_dir}/{dataset_name}.zarr.tar.lz4")
    subprocess.run(["rm", f"{shm_data_dir}/{dataset_name}.zarr.tar.lz4"], check=True)



def process_dataset(dataset_name: str, dataset_url: str, data_dir: str) -> None:
    if os.path.exists(f"{data_dir}/lz4/{dataset_name}.zarr.tar.lz4"):
        print(f"Skipping {dataset_name} because lz4 file already exists at {data_dir}/lz4/{dataset_name}.zarr.tar.lz4")
        return
    if not os.path.exists(f"{data_dir}/zip/{dataset_name}.zarr.zip"):
        download_data(dataset_name, dataset_url, f"{data_dir}/zip")
    else:
        print(f"Skipping {dataset_name} because zip file already exists at {data_dir}/zip/{dataset_name}.zarr.zip")
    convert_zip_to_lz4(dataset_name, f"{data_dir}/zip")

def copy_chunk(source_data, data_array, start, end):
    data_array[start:end] = source_data[start:end]

def remove_zarr_codec(dataset_name: str, data_dir: str) -> None:
    # This function has to be done in the umi conda environment
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
    register_codecs()

    source_dir = f"{data_dir}/{dataset_name}.zarr"
    os.makedirs(f"{data_dir}/lz4", exist_ok=True)
    target_dir = f"{data_dir}/lz4/{dataset_name}.zarr"

    source_group = zarr.open(source_dir, mode="r")
    assert isinstance(source_group, zarr.Group)
    source_meta_group = source_group["meta"]
    assert isinstance(source_meta_group, zarr.Group)
    source_data_group = source_group["data"]
    assert isinstance(source_data_group, zarr.Group)

    target_group = cast(zarr.Group, zarr.open(target_dir, mode="w"))
    target_meta_group = target_group.create_group("meta")
    target_data_group = target_group.create_group("data")

    for key in source_meta_group.keys():
        source_meta = cast(zarr.Array, source_meta_group[key])
        meta_array = target_meta_group.create_dataset(key, shape=source_meta.shape, dtype=source_meta.dtype)
        meta_array[:] = source_meta[:]


    for key in source_data_group.keys():
        source_data = cast(zarr.Array, source_data_group[key])
        print(f"key: {key}, {source_data.shape=}, {source_data.dtype=}, {source_data.compressor=}")
        if source_data.compressor is not None:
            
            chunk_size = (1, 224, 224, 3)
            # Parallelize the copying process
            data_array = target_data_group.create_dataset(key, shape=source_data.shape, chunks=chunk_size, dtype=source_data.dtype, compression="lz4")
            chunk_size = 1000  # Adjust this based on your data size and system capabilities
            num_chunks = (source_data.shape[0] + chunk_size - 1) // chunk_size
            with mp.Pool(mp.cpu_count() - 4) as pool:
                pool.starmap(copy_chunk, [(source_data, data_array, i * chunk_size, min((i + 1) * chunk_size, source_data.shape[0])) for i in range(num_chunks)])

        else:
            data_array = target_data_group.create_dataset(key, shape=source_data.shape, dtype=source_data.dtype, compression=None)
            data_array[:] = source_data[:]



if __name__ == "__main__":
    # Download the data
    # data_dir = "/scratch/m000073/uva/umi_data"
    # num_processes = mp.cpu_count()
    # with mp.Pool(num_processes) as pool:
    #     pool.starmap(process_dataset, [(dataset_name, url, data_dir) for dataset_name, url in DATASETS.items()])

    shm_data_dir = f"/dev/shm/{PROJECT_NAME}/umi_data"
    # remove_zarr_codec("cup_arrangement_1", f"{shm_data_dir}")
    remove_zarr_codec("cup_arrangement_1", f"{shm_data_dir}")
