%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Creates a connectivity matrix where the features maps connect to the single
% connections along the diagonal, followed by the double connections, tiple
% connections and so on until \c ydim is reached. 
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @conmat_file @copybrief conmat_someall_start_ones.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief conmat_someall_start_ones.m
%
% @param xdim This is first dimension of the connectivity matrix. This
% represents the number of input maps.
% @param ydim This is the second dimension fo the connectivity matrix. This
% represents the number of features maps.
% @retval C The specified connectivity matrix.
% @retval recommended The number of features maps that should be used given the
% requsted size and type of connectivity matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C,recommended] = conmat_someall_start_ones(xdim,ydim)

C = zeros(xdim,ydim);

% Special cases for only 1,2,or3 input maps (to make it random).
if(size(C,1) == 1)
    C = (randsample(2,size(C,2),true)-1)';
elseif(size(C,1) == 2)
    C(1,:) = (randsample(2,size(C,2),true)-1)';
    C(2,:) = (randsample(2,size(C,2),true)-1)';
elseif(size(C,1) == 3)
    % The number of elements
    connectivity_size = 0;
    % Where to start in each column.
    start_in_j = 0;
    % Number of times a connectivity_size was used.
    used_con = 0;
    % Input connectivity matrix.
    % For each feature map
    for k=1:size(C,2)
        % Prevent all ones from being in the column.
        if(connectivity_size==size(C,1)-1 && used_con>=1)
            C(:,k) = (randsample(2,size(C,1),true)-1);
        else
            % For each input map
            for j=1:size(C,1)
                % Suitable locations for the given location in j and size of
                % connectivity list.
                locations = mod(start_in_j:start_in_j+connectivity_size,size(C,1))+1;
                % If within the region for the current k, then set to 1.
                if(find(locations == j))
                    C(j,k) = 1;
                end
            end
            % Increment the number of times you've used this size.
            used_con = used_con+1;
            % Reached the end of the column using the current connectivity size
            if(used_con == size(C,1))
                connectivity_size = connectivity_size+1;
                used_con = 0;
            end
            % Increase the element that this size starts from.
            start_in_j = mod((start_in_j),size(C,1))+1;
        end
    end
else
    
    % The number of elements
    connectivity_size = 0;
    % Where to start in each column.
    start_in_j = 0;
    % Number of times a connectivity_size was used.
    used_con = 0;
    % Input connectivity matrix.
    % For each feature map
    for k=1:size(C,2)
        % Prevent all ones from being in the column.
        if(connectivity_size==size(C,1)-1 && used_con>=1)
            C(:,k) = (randsample(2,size(C,1),true)-1);
        else
            % For each input map
            for j=1:size(C,1)
                % Suitable locations for the given location in j and size of
                % connectivity list.
                locations = mod(start_in_j:start_in_j+connectivity_size,size(C,1))+1;
                % If within the region for the current k, then set to 1.
                if(find(locations == j))
                    C(j,k) = 1;
                end
            end
            % Increment the number of times you've used this size.
            used_con = used_con+1;
            % Reached the end of the column using the current connectivity size
            if(used_con == size(C,1))
                connectivity_size = connectivity_size+1;
                used_con = 0;
            end
            % Increase the element that this size starts from.
            start_in_j = mod((start_in_j),size(C,1))+1;
        end
    end
end


% Make sure there are no zero columns.
for k=1:size(C,2)
    % Keep randomly sampling to make sure each feature map is assigned to
    % atleast one input map.
   while(sum(C(:,k)) == 0) 
       C(:,k) = (randsample(2,size(C,1),true)-1);
   end
end


recommended = ydim;


end