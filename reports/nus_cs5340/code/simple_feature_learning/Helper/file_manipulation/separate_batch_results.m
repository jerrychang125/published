%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Separates the batch results into individual files so that I can use them
% in MyBuildPyramid example (just like Svetlana loads individual images).
% Also assumes they were all infered with the same set of filters.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inparam \li \e LAYER the layer to load \li \e RUN the run the files 
% where a part of. \li \e SAVE_RUN can save to a new run # \li \e image_path for
% both the test and train sets \li \e start_path where to put the results.
%
% @fileman_file @copybrief separate_batch_results.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% save the results? may just want ot have them left in worksapce.
SAVE_RESULTS = 1;

% LAYER TO LOAD (ONLY DOES ONE AT A TIME);
LAYER = 1;
% The run that all these were in.
RUN = 14;
% Save to a new run?
SAVE_RUN = 14;

% This path is used to get the naming of each file.
% This has to be the same size as the number of trained examples.
image_path = '/misc/FergusGroup/zeiler/Datasets/Caltech/101/150_max_set1/101_150_max_train1/';
[imfolders,trainfiles] = split_folders_files(dir(image_path));
image_path = '/misc/FergusGroup/zeiler/Datasets/Caltech/101/150_max_set1/101_150_max_test1/';
[imfolders,testfiles] = split_folders_files(dir(image_path));

% Start string that everything else will be appended to.
start_path = '/misc/FergusGroup/zeiler/Results/train/city_fruit/CN_gray_filters_8_32/Run_15/recon/101_150_max_';
% start_path = '/misc/FergusGroup/zeiler/Results/train/fruit_100_100/CN_color_filters_9_129/Run_1/recon/101_150_max_';

% Where to put the large concatenated results.

save_path = start_path;


% This includes the set number.
train_string = 'train1_';
test_string = 'test1_';
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
    
    % Index into the image.
    train_image = 0;
    test_image = 0;
    
    % Loop over each batch.
    for batch=1:10
        loadpath = strcat(start_path,train_string,batch_string,num2str(batch),'/Run_',num2str(RUN),'/')
        % Load the last epoch.
        trainepoch = get_highest_epoch(loadpath,layer);
        % Load the training data.
        load(strcat(loadpath,'epoch',num2str(trainepoch),'_layer',num2str(layer)));
        
        
        fprintf('Loaded batch: %d layer: %d\n',batch,layer);
%         fprintf('XXXX Note this is now normalizing z with svm_rescale2XXXXXXX\N')
%         z = svm_rescale2(z);
        % Conatenate the variables.
        Trainz = z;
        Trainy = y;
        Trainz0 = z0;
        TrainF = F;
        Trainmodel = model;
        
        
        % For each image save the individual case.
        for image=1:size(Trainz,4)
            train_image = train_image+1;
            z = Trainz(:,:,:,image);
            y = Trainy(:,:,:,image);
            try
                z0 = Trainz0(:,:,:,image);
            catch
                z0 = [];
            end
            model = Trainmodel;
            F = TrainF;
            
            [a,name,c,d] = fileparts(trainfiles(train_image).name);
            trainsavepath = strcat(save_path,train_string,'separated/Run_',num2str(SAVE_RUN),'/epoch',...
                num2str(trainepoch),'_layer',num2str(layer),'/',name,'.mat');
            fprintf('Saving Train (%d/%d): %s\n',train_image,length(trainfiles),trainsavepath);
            
            if(image==1)
                mkdir(parentdir(trainsavepath));
            end
                        save(trainsavepath,'y','z','F','z0','model','-v7.3');
        end
        
        
        loadpath = strcat(start_path,test_string,batch_string,num2str(batch),'/Run_',num2str(RUN),'/')
        % Load the last epoch.
        testepoch = get_highest_epoch(loadpath,layer);
        % Load the training data.
        load(strcat(loadpath,'epoch',num2str(testepoch),'_layer',num2str(layer)));
        fprintf('Test loaded batch: %d layer: %d\n',batch,layer);
        
        % Conatenate the variables.
%                 fprintf('XXXX Note this is now normalizing z with svm_rescale2XXXXXXX\N')
%         z = svm_rescale2(z);
        Testz = z;
        Testy = y;
        Testz0 = z0;
        TestF = F;
        Testmodel = model;
        size(Testz)
        
        % For each image save the individual case.
        for image=1:size(Testz,4)
            test_image = test_image+1;
            z = Testz(:,:,:,image);
            y = Testy(:,:,:,image);
            try
                z0 = Testz0(:,:,:,image);
            catch
                z0 = [];
            end
            model = Testmodel;
            F = TestF;
            
            [a,name,c,d] = fileparts(testfiles(test_image).name);
            testsavepath = strcat(save_path,test_string,'separated/Run_',num2str(SAVE_RUN),'/epoch',...
                num2str(testepoch),'_layer',num2str(layer),'/',name,'.mat');
            fprintf('Saving Test (%d/%d): %s\n',test_image,length(testfiles),testsavepath);
            if(image==1)
                mkdir(parentdir(testsavepath));
            end
                        save(testsavepath,'y','z','F','z0','model','-v7.3');
        end
    end
end