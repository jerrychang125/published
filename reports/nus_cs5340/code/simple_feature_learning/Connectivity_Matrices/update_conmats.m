%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates any type of connectivity matrix using the conmat_... files. The type
% and size of each is defined by the model struct and the generated connectivity
% matrices will be returned in those. This can also be used to update the gui
% components for the number of feature maps (if they are passed in). If they are
% not passed in, those return values are [].
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief update_conmats.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief update_conmats.m
%
% @param model the model struct that must have the conmat_types, num_layers,
% num_input_maps, num_feature_maps, and conmats fields initialized.
% @param hnumfeatures the gui component for the number of feature maps in layer
% 1.
% @param hnumfeatures2 the gui component for the number of feature maps in layer
% 2.
% @param hnumfeatures3 the gui component for the number of feature maps in layer
% 3.
% @param hnumfeatures4 the gui component for the number of feature maps in layer
% 4.
% @retval model updated model structure with the conmats in it.
% @retval hnumfeautres the updated gui component for layer 1 (or [] if none was
% passed in).
% @retval hnumfeautres2 the updated gui component for layer 2 (or [] if none was
% passed in).
% @retval hnumfeautres3 the updated gui component for layer 3 (or [] if none was
% passed in).
% @retval hnumfeautres4` the updated gui component for layer 4 (or [] if none was
% passed in).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model,hnumfeatures,hnumfeatures2,hnumfeatures3,hnumfeatures4] = update_conmats(model,hnumfeatures,hnumfeatures2,hnumfeatures3,hnumfeatures4)

% If the gui components are not passed in, then just return nothing in
% them.
if(nargin == 1)
    hnumfeatures = [];
    hnumfeatures2 = [];
    hnumfeatures3 = [];
    hnumfeatures4 = [];
end


for layer=1:model.num_layers
    
    % Set current data to the selected data set.
    switch model.conmat_types{layer}
        case 'Full' % User selects Peaks.
            [temp_conmat,recommended] = conmat_full(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles' % User selects Membrane.
            [temp_conmat,recommended] = conmat_singles(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles and Some Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_somedoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles and All Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_alldoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles and Random Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_randdoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Some Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_somedoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'All Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_alldoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Random Doubles' % User selects Sinc.
            [temp_conmat,recommended] = conmat_randdoub(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Some Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_sometrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'All Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_alltrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Random Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_randtrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, Some Doubles, and Some Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_somedoub_sometrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, All Doubles, and Some Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_alldoub_sometrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, Random Doubles, and Some Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_randdoub_sometrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, Some Doubles, and Random Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_somedoub_randtrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, All Doubles, and Random Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_alldoub_randtrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, All Doubles, and All Triples' % User selects Sinc.
            [temp_conmat,recommended] = conmat_singles_alldoub_alltrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Singles, Random Doubles, and Random Triples' % User selects Sinc
            [temp_conmat,recommended] = conmat_singles_randdoub_randtrip(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Some of each (starting with singles)' % User selects Sinc.
            [temp_conmat,recommended] = conmat_someall_start_ones(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Some of each (starting with doubles)' % User selects Sinc.
            [temp_conmat,recommended] = conmat_someall_start_twos(model.num_input_maps(layer),model.num_feature_maps(layer));
        case 'Even Number of Rand Sing,Doub,Trip'
            [temp_conmat,recommended] = conmat_randsing_randdoub_randtrip(model.num_input_maps(layer),model.num_feature_maps(layer));
    end
    
    
    if(nargin ~= 1)
        % Update the number of features in the gui.
        switch layer
            case 1
                set(hnumfeatures,'Value',recommended);
                set(hnumfeatures,'String',num2str(recommended));
            case 2
                set(hnumfeatures2,'Value',recommended);
                set(hnumfeatures2,'String',num2str(recommended));
            case 3
                set(hnumfeatures3,'Value',recommended);
                set(hnumfeatures3,'String',num2str(recommended));
            case 4
                set(hnumfeatures4,'Value',recommended);
                set(hnumfeatures4,'String',num2str(recommended));
        end
    end
    
    % Save the conmat and the num_feature_maps.
    model.conmats{layer} = temp_conmat;
    model.num_feature_maps(layer) = recommended;
    fprintf('New Connectivity Matrix for Layer %d is: \n',layer);
    disp(model.conmats{layer})
    if(layer<model.num_layers)
        model.num_input_maps(layer+1) = recommended;
    end
end
end