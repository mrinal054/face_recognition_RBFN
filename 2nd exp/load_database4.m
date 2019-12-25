function out=load_database4();
% We load the database the first time we run the program.

persistent loaded;
persistent y;
if(isempty(loaded))
    v=zeros(10304,200);
    for i=1:40
        cd(strcat('t',num2str(i)));
        for j=1:5
            a=imread(strcat(num2str(j),'.pgm'));
          
            v(:,(i-1)*5+j)=reshape(a,size(a,1)*size(a,2),1);
        end
        cd ..
    end
    y=uint8(v); % Convert to unsigned 8 bit numbers to save memory. 
end
loaded=1;  % Set 'loaded' to aviod loading the database again. 
out=y;