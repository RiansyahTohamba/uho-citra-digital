import yaml

class Konfigurasi():
    YML_FILE = "config.yml"

    def read_yaml(self):
        with open(self.YML_FILE, "r") as file:
            return yaml.safe_load(file)

    def update_yaml(self,key, value):
        data = self.read_yaml()
        keys = key.split('.')  # Mendukung nested key (misalnya "settings.theme")
        
        # Navigasi ke key yang diinginkan
        temp = data
        for k in keys[:-1]:
            temp = temp.setdefault(k, {})
        
        temp[keys[-1]] = value  # Update nilai
        
        with open(self.YML_FILE, "w") as file:
            yaml.safe_dump(data, file)
