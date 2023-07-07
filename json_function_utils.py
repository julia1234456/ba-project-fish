import json
import matplotlib.pyplot as plt


def rajouter_virgule(fichier_json):
    with open(fichier_json, 'r') as f:
        contenu = f.read()

    contenu_modifie = contenu.replace('}', '},')

    with open(fichier_json, 'w') as f:
        f.write(contenu_modifie)
        
def remove_last_comma(json_data):
    last_comma_index = json_data.rfind(',')
    if last_comma_index != -1:
        json_data = json_data[:last_comma_index] + json_data[last_comma_index+1:]
    return json_data

ef add_commas_and_brackets_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read()
    
    data_modif = data.replace('}', '},')
    data_modif = '[' + ''.join(data_modif) + ']'
    data_modif = remove_last_comma(data_modif)
    
    with open(output_file, 'w') as f:
        f.write(data_modif)

# Exemple d'utilisation
input_file = '5fish_10_000it.json'
output_file = '5fish_10_000it.json'
add_commas_and_brackets_to_json(input_file, output_file)


def create_total_loss_graph(fichier_json):
    with open(fichier_json) as f:
        data = json.load(f)

    field1 = []
    field2 = []
    for item in data:
        if 'total_loss' in item :
            field1.append(item['iteration'])
            field2.append(item['total_loss'])


    plt.plot(field1, field2, '.')
 
    plt.xlabel('iteration')
    plt.ylabel('total loss')
    plt.title('Evolution of the total loss in fonction of the number of iterations')
    plt.show()