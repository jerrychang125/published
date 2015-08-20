%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Collects all the feature maps that are saved for each batch into a single
% large file (all assumed to ahve the same three dimensions).
% Also assumes they were all infered with the same set of filters.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inparam \li \e LAYER the specific layer to load
% \li \e RUN the run these files were in \li start_path the path to the maps
%
% @fileman_file @copybrief collect_batch_results.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% save the results? may just want ot have them left in worksapce.
SAVE_RESULTS = 1;

% LAYER TO LOAD (ONLY DOES ONE AT A TIME);
LAYER = 2;
% The run that all these were in.
RUN = 1;

% Start string that everything else will be appended to.
start_path = '/misc/FergusGroup/zeiler/Results/train/city_fruit/CN_gray_filters_8_32/Run_15/recon/101_200_max_';
% Where to put the large concatenated results.
save_path = start_path;

% This includes the set number.
train_string = 'train11_';
test_string = 'test11_';
% The string to append before the batch number.
batch_string = 'batch';


for layer=1:LAYER
    
    % The total variables.
    Trainz = [];
    Trainy = [];
    Trainz0 = [];
    % The total variables.
    Testz = [];
    Testy = [];
    Testz0 = [];
    
    % Loop over each batch.
    for batch=1:10
        loadpath = strcat(start_path,train_string,batch_string,num2str(batch),'/Run_',num2str(RUN),'/')
        % Load the last epoch.
        trainepoch = get_highest_epoch(loadpath,layer);
        % Load the training data.
        load(strcat(loadpath,'epoch',num2str(trainepoch),'_layer',num2str(layer)));
        
        fprintf('Loaded batch: %d layer: %d\n',batch,layer);
        % Conatenate the variables.
        Trainz = cat(4,Trainz,z);
        Trainy = cat(4,Trainy,y);
        Trainz0 = cat(4,Trainz0,z0);
        size(Trainz)
        
        loadpath = strcat(start_path,test_string,batch_string,num2str(batch),'/Run_',num2str(RUN),'/');
        % Load the last epoch.
        testepoch = get_highest_epoch(loadpath,layer);
        % Load the training data.
        load(strcat(loadpath,'epoch',num2str(testepoch),'_layer',num2str(layer)));
        fprintf('Test loaded batch: %d layer: %d\n',batch,layer);
        
        % Conatenate the variables.
        Testz = cat(4,Testz,z);
        Testy = cat(4,Testy,y);
        Testz0 = cat(4,Testz0,z0);
        size(Testz)
        
    end
    
    
    clear y z z0
    
    % Make the save variables ready.
    y = Trainy;
    z = Trainz;
    z0 = Trainz0;
    
    % Save the training data.
    if(SAVE_RESULTS)
        trainsavepath = strcat(save_path,train_string,'combined/Run_',num2str(RUN),'/epoch',...
            num2str(trainepoch),'_layer',num2str(layer),'.mat')
        mkdir(parentdir(trainsavepath));
        save(trainsavepath,'y','z','F','z0','model','-v7.3');
    end
    
    
    clear y z z0
    
    % Make the save variables ready.
    y = Testy;
    z = Testz;
    z0 = Testz0;
    
    % Save the training data.
    if(SAVE_RESULTS)
        testsavepath = strcat(save_path,test_string,'combined/Run_',num2str(RUN),'/epoch',...
            num2str(testepoch),'_layer',num2str(layer),'.mat')
        mkdir(parentdir(testsavepath));
        save(trainsavepath,'y','z','F','z0','model','-v7.3');
    end
end