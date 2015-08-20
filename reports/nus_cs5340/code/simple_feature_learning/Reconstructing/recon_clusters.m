%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% File to launch multiple recon jobs on separate machines.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @recon_file @copybrief recon_clusters.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup the model parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the parameters for the experiment.
clear all

% Load the gui parameters.
load('gui_has_set_the_params.mat')
format compact

% The computation threads used for each job.
COMP_THREADS = 2;

% Set number of computation threads to use.
maxNumCompThreads(model.comp_threads);
%%%%%%%%%%

% Make sure these numbers add up to exactly 20.
machine_list = {
     % The first one has to be >= 1 as well (will always put a job on the first machine regardless of this number).
    {' zeiler@greendotblade3.cs.nyu.edu',4},...
    {' zeiler@greendotblade1.cs.nyu.edu',5},...
    {' zeiler@greendotblade4.cs.nyu.edu',1},...
    {' zeiler@ceres.cs.nyu.edu',2},...
    {' zeiler@iris.cs.nyu.edu',3},...
    {' zeiler@juno.cs.nyu.edu',5},...
    {' zeiler@hamlet.cs.nyu.edu',0},...
    {' zeiler@blakey.cs.nyu.edu',0},...
    {' zeiler@django.cs.nyu.edu',0}};



% The images to infer up.
origfulldatapath = model.fulldatapath
% Where to save
origfullsavepath = model.fullsavepath

machineInd = 1;

count_on_machine = 0;
% jobs_per_machine = ceil(20/length(machine_list))
jobs_per_machine = 8;
for i=1:2
    
    
    % Loop over each batch.
    for batch=1:10
        
        
        %%%%%%%%%
        %% Set parameters for each separate job.
        %%%%%%%%%
        model.comp_threads = COMP_THREADS;
        fprintf('Current jobs paths:\n');
        
        % Replace the test with train names in the model string.
        if(i==2)
            fulldatapath = regexprep(origfulldatapath,'_train','_test');
            fullsavepath = regexprep(origfullsavepath,'_train','_test');
        else
            fulldatapath = origfulldatapath;
            fullsavepath = origfullsavepath;
        end
        fulldatapath = regexprep(fulldatapath,'batch(\d*)/',strcat('batch',num2str(batch),'/'));
        fullsavepath = regexprep(fullsavepath,'batch(\d*)/',strcat('batch',num2str(batch),'/'));
        fprintf('fulldatapath: %s\n',fulldatapath);
        fprintf('fullsavepath: %s\n',fullsavepath);
        model.fulldatapath = fulldatapath;
        
        model.fullsavepath = fullsavepath;
        
        % Make sure the direcotry exists for the save path.
        mkdir(fullsavepath);
        %%%%%%%%%
        
        
        
        % Have to give it time to launch window and load gui_has_set_the_params.mat
        user_entry = input('Has the job loaded the images yet and (s for switch after this job)? (y/s/n):\n','s');
        if(strcmp(user_entry,'y') || strcmp(user_entry,'yes'))
            fprintf('Starting on machine %s (%d out of %d) and now running this job (%d/%d) on that machine.\n\n',machine_list{machineInd}{1},machineInd,length(machine_list),count_on_machine+1,machine_list{machineInd}{2});            
            
            
            
                    %%%%%%%%%
                    %% Save the new gui_has_set_the_params.mat file
                    %%%%%%%%%
                    save('/home/zeiler/work/Thesis/A/GUI/gui_has_set_the_params.mat','model');
            
            
                    %%%%%%%%%%
                    %% Launch a job on a new server
                    %%%%%%%%%%
                    machine_temp = machine_list{machineInd}{1};
                    eval(strcat('! xterm -T ',machine_temp,'-',model.fullsavepath(max(0,length(model.fullsavepath)-50):end),...
                        ' -e bash /home/zeiler/work/Thesis/A/GUI/start_server_java.sh ',machine_temp,' ',model.expfile,' &'))
            
            
            % Change machines.
            count_on_machine = count_on_machine + 1;
            % If on the last machine then it will just fill it up.
            while(count_on_machine == machine_list{machineInd}{2} && machineInd ~= length(machine_list))
                count_on_machine = 0;
                machineInd = machineInd + 1; % Go to next machine.
            fprintf('Auto-moving to machine %s (%d out of %d) for next job.\n\n',machine_list{machineInd}{1},machineInd,length(machine_list));            
                
            end
%         elseif(strcmp(user_entry,'s')) % If you want to switch machines.
%             machineInd = machineInd + 1;
%             count_on_machine = 1;
%             fprintf('Moving to machine %s (%d out of %d) and now running this job (%d/%d) on that machine.\n\n',machine_list{machineInd},machineInd,length(machine_list),count_on_machine,jobs_per_machine);            
%             
%             
            
                        
%                     %%%%%%%%%
%                     %% Save the new gui_has_set_the_params.mat file
%                     %%%%%%%%%
%                     save('/home/zeiler/work/Thesis/A/GUI/gui_has_set_the_params.mat','model');
%             
%             
%                     %%%%%%%%%%
%                     %% Launch a job on a new server
%                     %%%%%%%%%%
%                     machine_temp = machine_list{machineInd};
%                     eval(strcat('! xterm -T ',machine_temp,'-',model.fullsavepath(max(0,length(model.fullsavepath)-50):end),...
%                         ' -e bash /home/zeiler/work/Thesis/A/GUI/start_server_java.sh ',machine_list{machineInd},' ',model.expfile,' &'))
%             
%             
%             

            
            
            
            
            
        else
            break
        end
    end
end
    
