

dataset = ['ml100k', 'mlls', 'CSJ', 'Ciao', 'PP', 'ml1m', 'modcloth']
modes = ['explicit', 'implicit']

data = {

    'ml100k':{
        'userCount':943,
        'itemCount':1682,
        'url':'../Data/ml100k/initial/ml100k',
        'split':'\t',
    },

    'mlls':{
        'userCount':610,
        'itemCount':9724,
        'url':'../Data/mlls/initial/mlls',
        'split':'\t',
    },

    'CSJ':{
        'userCount':39387,
        'itemCount':23033,
        'url':'../Data/CSJ/initial/csj',
        'split':'\t',
    },

    'Ciao':{
        'userCount':17615,
        'itemCount':16121,
        'url':'../Data/Ciao/initial/Ciao',
        'split':' ',
    },

    'PP':{
        'userCount':14180,
        'itemCount':4970,
        'url':'../Data/PP/initial/PP',
        'split':'\t',
    },
    'ml1m':{
        'userCount':6040,
        'itemCount':3706,
        'url':'../Data/ml1m/initial/ml1m',
        'split':'\t',
    },
    'modcloth':{
        'userCount':2360,
        'itemCount':634,
        'url':'../Data/modcloth/initial/modcloth',
        'split':',',
    }

}

methods = {

    'MPS': ['initial', 'FI', 'FI_nosim'],
    'FPS': ['Del', 'RMZ', 'Trust', 'randomDel'],
    'PPS':['del_test', 'reserve_test', 'nsps', 'random'],

}





