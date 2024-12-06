local dap = require('dap')
dap.adapters.python = {
    type = 'executable',
    command = 'python',
    args = {'-m', 'debugpy.adapter'}
}

dap.configurations.python = {
    {
        type = 'python',
        request = 'launch',
        name = 'Launch file',
        program = '${file}'
    },
    {
        type = 'python',
        request = 'launch',
        name = 'Train Multi-Agent',
        program = '${file}',
        args = {'--train_agent', 
            '--resume', 
            'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', 
            '--data-path', 
            '../../imagenet'
        }
    }
}
