import yaml

with open("test.yaml", "r") as stream:
    try: 
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
    
print(data)