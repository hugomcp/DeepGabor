function ret=get_Gabor_filters(wavelengthGabor,orientationGabor,phaseGabor,ratioGabor)
%29-09-2011
%Returns a set of Gabor filters with all the combinations of parameters
%Parameter 1: 'wavelengthGabor'
%Parameter 2: 'orientationGabor'
%Parameter 3: 'phaseGabor'
%Parameter 4: 'ratioGabor'


sigmaGabor=0.5*wavelengthGabor;

ret=cell(numel(wavelengthGabor)*numel(orientationGabor)*numel(phaseGabor)*numel(ratioGabor),3); 
    %Col 1: Real Part, 2: Imaginary Part, 3=Configs string

index=1;
for w=1:numel(wavelengthGabor)
    for o=1:numel(orientationGabor)
        for p=1:numel(phaseGabor)
            for r=1:numel(ratioGabor)
                
                
                sigma_x = sigmaGabor(w);
                sigma_y = sigmaGabor(w)/ratioGabor(r);
                
                
                % Bounding box
                nstds = 3;
                xmax = max(abs(nstds*sigma_x*cos(orientationGabor(o))),abs(nstds*sigma_y*sin(orientationGabor(o))));
                xmax = ceil(max(1,xmax));
                ymax = max(abs(nstds*sigma_x*sin(orientationGabor(o))),abs(nstds*sigma_y*cos(orientationGabor(o))));
                ymax = ceil(max(1,ymax));
                xmin = -xmax; ymin = -ymax;
                [x,y] = meshgrid(xmin:xmax,ymin:ymax);                

                
                % Rotation
                x_theta=x*cos(orientationGabor(o))+y*sin(orientationGabor(o));
                y_theta=-x*sin(orientationGabor(o))+y*cos(orientationGabor(o));
                                
                ret{index,1}= 1/(2*pi*sigma_x *sigma_y) *...
                    exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/wavelengthGabor(w)*x_theta+phaseGabor(p));
                ret{index,2}= 1/(2*pi*sigma_x *sigma_y) *...
                    exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*sin(2*pi/wavelengthGabor(w)*x_theta+phaseGabor(p));
                
                %remove filter energy
                ret{index,1}=ret{index,1}-mean(ret{index,1}(:));
                ret{index,2}=ret{index,2}-mean(ret{index,2}(:));
                
                ret{index,3}=['w:',num2str(wavelengthGabor(w)),'o:',num2str(orientationGabor(o)),...
                    'p:',num2str(phaseGabor(p)),'r:',num2str(ratioGabor(r))];
                 
%                subplot(1,2,1), imshow(ret{index,1},[]), title(['Real ',num2str(size(ret{index,1}))]),...
%                    subplot(1,2,2), imshow(ret{index,2},[]), title (['Imaginary ',num2str(size(ret{index,1}))]);
                
%                waitforbuttonpress;
                

                index=index+1;
            end
        end
    end
end






