import os
import unittest
import requests

def iterate_files_in_directories(directory) -> iter:
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

class DownloadGaiai(unittest.TestCase):
    def test_something(self):

        for f in iterate_files_in_directories("/Users/hayde/Downloads/gaia"):
            with open(os.path.join("/Users/hayde/Downloads/gaia", f), 'r') as file_to_read:
                for l in file_to_read.readlines():
                    l = l.split("  ")[1].strip()
                    # if "gdr1" in f:
                    #     r = f"https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/{l}"
                    #     print("Downloading", r)
                    #     self.do_req(l, r, "gdr1")
                    if "gdr2" in f:
                        r = f"https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/{l}"
                        print("Downloading", r)
                        self.do_req(l, r, "gdr2")
                    if "gdr3" in f:
                        r = f"https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/{l}"
                        print("Downloading", r)
                        self.do_req(l, r, "gdr3")
                    if "gedr3" in f:
                        r = f"https://cdn.gea.esac.esa.int/Gaia/gedr3/gaia_source/{l}"
                        print("Downloading", r)
                        self.do_req(l, r, "gedr3")
                    if "gfpr" in f:
                        r = f"https://cdn.gea.esac.esa.int/Gaia/gfpr/Solar_system/sso_source/{l}"
                        print("Downloading", r)
                        self.do_req(l, r, "gfpr")

    def do_req(self, l, r, sub):
        if os.path.exists(os.path.join(f"/Volumes/18/gaia/{sub}/{l}")):
            return
        res = requests.get(url=r, stream=True, verify=False)
        if res.status_code == 200:
            with open(f"/Volumes/18/gaia/{sub}/{l}", 'wb') as gaia_res:
                for chunk in res:
                    gaia_res.write(chunk)
        else:
            print(res)


if __name__ == '__main__':
    unittest.main()
