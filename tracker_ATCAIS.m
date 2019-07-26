
tracker_label = 'ATCAIS';


tracker_command = generate_python_command('vot_ATCAIS', ...
    {'/home/tangjiuqi097/vot/ATCAIS/pytracking/vot', ...  % tracker source and vot.py are here
    '/home/tangjiuqi097/vot/ATCAIS/pytracking', ...
    '/home/tangjiuqi097/vot/ATCAIS/', ...
    '/home/tangjiuqi097/data/vot2019/vot-toolkit_7_0_2/native/trax/support/python'
    });


tracker_interpreter = 'python';

tracker_linkpath = {'/home/tangjiuqi097/data/vot2019/vot-toolkit_7_0_2/native/trax/build'};
