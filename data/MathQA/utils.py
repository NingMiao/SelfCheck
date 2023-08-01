def get_option_id(selecting_output):
    selection_dict={'a )': 'a', 'a)': 'a',
                    'b )': 'b', 'b)': 'b',
                    'c )': 'c', 'c)': 'c',
                    'd )': 'd', 'd)': 'd',
                    'e )': 'e', 'e)': 'e',}
    
    selecting_output=selecting_output.lower()
    for key in selection_dict:
        if key in selecting_output:
            return selection_dict[key]
    return None