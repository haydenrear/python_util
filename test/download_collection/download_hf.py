import os
import sys
import time
import unittest
from threading import Thread

import psutil
from huggingface_hub import HfApi, HfFolder, snapshot_download, ModelFilter


def iterate_files_in_directories(directory) -> iter:
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


class DownloadHfCollection(unittest.TestCase):
    def test_do_download(self):
        # Replace 'your_organization_name' with the name of your Hugging Face organization
        organization_name = 'yuchen0187'
        to_save_volume = '/Volumes/16'

        # Define a directory to save the downloaded models

        def do_download():
            # Initialize the Hugging Face API
            api = HfApi()

            # Fetch the list of models in the organization
            models = api.list_models(filter=ModelFilter(author=organization_name))
            print(models)
            for model in models:
                model_id = model.modelId
                model_id = str(model_id)
                print(f'Downloading model: {model_id}')
                # Download the model to the specified directory
                print(f'Downloading {str(model_id)}.')
                replace = str(model_id).replace(f'{organization_name}/', '')
                try:
                    if not model_id.startswith(f'{organization_name}/') or not 'Point-SAM' in model_id:
                        print("Skipping ", model_id)
                        continue
                    model_name = f'models--{organization_name}--{replace}'
                    did_find = False
                    print("Testing if exists already ", model_name)
                    for m in iterate_files_in_directories(
                            os.path.join("/Users/hayde/IdeaProjects/ml/models-scripts", "indices")):
                        with open(m, 'r') as f:
                            if any([model_name == potential_model_name.strip() for potential_model_name in
                                    f.readlines()]):
                                print(f"found model {model_name} already downloaded!")
                                did_find = True
                                break
                    if did_find:
                        continue
                    if not any([os.path.exists(f"/Volumes/{v}/models--{organization_name}--{replace}") for v in
                                ['12', '13', '14', '15', '16', '17', '18', '19', '20']]):
                        snapshot_download(repo_id=model_id, cache_dir=to_save_volume, token='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')
                        print(f'Model {str(model_id)} downloaded successfully.')
                    else:
                        print("Skipping ", model_id)
                except Exception as e:
                    print(e)
                    print("Deleting ", replace)
                    os.rmdir(f"{to_save_volume}/models--{organization_name}--{replace}")
                    p = psutil.disk_usage(to_save_volume).percent
                    if p <= 5:
                        sys.exit()

        t = Thread(target=do_download)
        t.start()
        t.join()


if __name__ == '__main__':
    unittest.main()
