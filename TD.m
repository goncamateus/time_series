function varargout = TD(varargin)
            % Copyright (c) 2010, Guillaume MAZE
            % All rights reserved.
            %
            % Redistribution and use in source and binary forms, with or without
            % modification, are permitted provided that the following conditions are
            % met:
            %
            %     * Redistributions of source code must retain the above copyright
            %       notice, this list of conditions and the following disclaimer.
            %     * Redistributions in binary form must reproduce the above copyright
            %       notice, this list of conditions and the following disclaimer in
            %       the documentation and/or other materials provided with the distribution
            %
            % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
            % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
            % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
            % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
            % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
            % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
            % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
            % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
            % POSSIBILITY OF SUCH DAMAGE.
            
            
            if nargin == 0
                disp_optionslist;
                return
            else
                narg = nargin - 7;
                if mod(narg,2) ~=0
                    error('taylordiag.m : Wrong number of arguments')
                end
            end
            
            STDs = varargin{1};
            RMSs = varargin{2};
            CORs = varargin{3};
            Pn = varargin{4};
            colors = varargin{5};
            plotRMS = varargin{6};
            plotSS4 = varargin{7};
            
            
            %% CHECK THE INPUT FIELDS:
            % apro = 100;
            % di   = fix(RMSs*apro)/apro - fix(sqrt(STDs.^2 + STDs(1)^2 - 2*STDs*STDs(1).*CORs)*apro)/apro;
            % if find(di~=0)
            % %	help taylordiag.m
            % 	ii = find(di~=0);
            % 	if length(ii) == length(di)
            % 		error(sprintf('taylordiag.m : Something''s wrong with ALL the datas\nYou must have:\nRMSs - sqrt(STDs.^2 + STDs(1)^2 - 2*STDs*STDs(1).*CORs) = 0 !'))
            % 	else
            % 		error(sprintf('taylordiag.m : Something''s wrong with data indice(s): [%i]\nYou must have:\nRMSs - sqrt(STDs.^2 + STDs(1)^2 - 2*STDs*STDs(1).*CORs) = 0 !',ii))
            % 	end
            % end
            
            %% IN POLAR COORDINATES:
            rho   = STDs;
            theta = real(acos(CORs));
            dx    = rho(1);	% Observed STD
            
            %%
            
            
            
            %% BEGIN THE PLOT HERE TO GET AXIS VALUES:
            hold off
            cax = gca;
            tc = get(cax,'xcolor');
            %ls = get(cax,'gridlinestyle');
            ls = '-'; % DEFINE HERE THE GRID STYLE
            next = lower(get(cax,'NextPlot'));
            
            %% LOAD CUSTOM OPTION OF AXE LIMIT:
            nopt = narg/2; foundrmax = 0;
            for iopt = 4 : 2 : narg+3
                optvalue = varargin{iopt+1};
                switch lower(varargin{iopt}), case 'limstd', rmax = optvalue; foundrmax=1; end
            end
            
            % make a radial grid
            hold(cax,'on');
            if foundrmax==0
                maxrho = max(abs(rho(:)));
            else
                maxrho = rmax;
            end
            hhh = line([-maxrho -maxrho maxrho maxrho],[-maxrho maxrho maxrho -maxrho],'parent',cax);
            set(cax,'dataaspectratio',[1 1 1],'plotboxaspectratiomode','auto')
            v = [get(cax,'xlim') get(cax,'ylim')];
            ticks = sum(get(cax,'ytick')>=0);
            delete(hhh);
            
            
            % check radial limits and ticks
            rmin = 0;
            smin = 0;
            if foundrmax == 0;
                rmax = v(4);
            end
            
            if rmax < 1.25*STDs(1)
                rmax = fix((1.25*STDs(1) + 0.1)*100)/100;
            end
            
            rticks = max(ticks-1,2);
            if rticks > 5   % see if we can reduce the number
                if rem(rticks,2) == 0
                    rticks = rticks/2;
                elseif rem(rticks,3) == 0
                    rticks = rticks/3;
                end
            end
            stdr = rmax/dx;
            smax  = max(((1+CORs(2:end)).^4)/(4*(stdr+1/stdr)^2));
            rinc  = (rmax-rmin)/rticks;
            sinc  = (smax-smin)/rticks;
            tick  = (rmin+rinc):rinc:rmax;
            tick_s = (smin+sinc):sinc:smax;
            
            %% LOAD DEFAULT PARAMETERS:
            if find(CORs<0)
                Npan = 2; % double panel
            else
                Npan = 1;
            end
            tickRMSangle  = 135;
            showlabelsRMS = 1;
            showlabelsSTD = 1;
            showlabelsCOR = 1;
            colSTD = [0 0 0];
            colRMS = [0 .6 0];
            colCOR = [0 0 1];
            tickCOR(1).val = [1 .99 .95 .9:-.1:0];
            tickCOR(2).val = [1 .99 .95 .9:-.1:0 -.1:-.1:-.9 -.95 -.99 -1];
            %             tickCOR(1).val = fliplr([0:0.1:1]);
            %             tickCOR(2).val = fliplr([-1:0.1:1]);
            widthCOR = .8;
            widthRMS = .8;
            widthSTD = .8;
            styleCOR = '-.';
            styleRMS = '--';
            styleSTD = ':';
            titleRMS = 1;
            titleCOR = 1;
            titleSTD = 1;
            tickRMS = tick; rincRMS = rinc;
            tickSTD = tick; rincSTD = rinc;
            tickSS4 = tick_s; %rincSS4 = sinc;
            
            
            %% LOAD CUSTOM OPTIONS:
            
            
            
            %CUSTOM FONT & FONTSIZE
            
            fNM = 'default';
            fSZ = 13;
            
            CFZ.std = fSZ;
            CFZ.rmsd = fSZ;
            CFZ.corr = fSZ;
            
            LAB.std = fSZ;
            LAB.rmsd = fSZ;
            LAB.corr = fSZ;
            
            % CUSTOM STD CIRCLES LABEL POSITION
            POS.std = 'bottom';
            
            nopt = narg/2;
            for iopt = 4 : 2 : narg+3
                optname  = varargin{iopt};
                optvalue = varargin{iopt+1};
                switch lower(optname)
                    
                    case 'tickrms'
                        tickRMS = sort(optvalue);
                        rincRMS = (max(tickRMS)-min(tickRMS))/length(tickRMS);
                    case 'showlabelsrms'
                        showlabelsRMS = optvalue;
                    case 'tickrmsangle'
                        tickRMSangle = optvalue;
                    case 'colrms'
                        colRMS = optvalue;
                    case 'widthrms'
                        widthRMS = optvalue;
                    case 'stylerms'
                        styleRMS = optvalue;
                    case 'titlerms'
                        titleRMS = optvalue;
                        
                    case 'tickstd'
                        tickSTD = sort(optvalue);
                        rincSTD = (max(tickSTD)-min(tickSTD))/length(tickSTD);
                    case 'showlabelsstd'
                        showlabelsSTD = optvalue;
                    case 'colstd'
                        colstd = optvalue;
                    case 'widthstd'
                        widthSTD = optvalue;
                    case 'stylestd'
                        styleSTD = optvalue;
                    case 'titlestd'
                        titleSTD = optvalue;
                    case 'npan'
                        Npan = optvalue;
                        
                    case 'tickcor'
                        tickCOR(Npan).val = optvalue;
                    case 'colcor'
                        colCOR = optvalue;
                    case 'widthcor'
                        widthCOR = optvalue;
                    case 'stylecor'
                        styleCOR = optvalue;
                    case 'titlecor'
                        titleCOR = optvalue;
                    case 'showlabelscor'
                        showlabelsCOR = optvalue;
                end
            end
            
            
            %% CONTINUE THE PLOT WITH UPDATED OPTIONS:
            
            % define a circle
            th = 0:pi/150:2*pi;
            xunit = cos(th);
            yunit = sin(th);
            % now really force points on x/y axes to lie on them exactly
            inds = 1:(length(th)-1)/4:length(th);
            xunit(inds(2:2:4)) = zeros(2,1);
            yunit(inds(1:2:5)) = zeros(3,1);
            % plot background if necessary
            if ~ischar(get(cax,'color')),
                %		ig = find(th>=0 & th<=pi);
                ig = 1:length(th);
                patch('xdata',xunit(ig)*rmax,'ydata',yunit(ig)*rmax, ...
                    'edgecolor',tc,'facecolor',get(cax,'color'),...
                    'handlevisibility','off','parent',cax);
            end
            
            % DRAW RMS & SS4 CURVES:
            
%             plotRMS = 1;
%             plotSS4 = 1;
            
            if plotSS4
                %%% ss4
                
%                 ((1+CORs).^4)./(4*(STDs/STDs(1)+STDs(1)./STDs).^2)

                ss4curves = [0.25:0.25:0.75,0.9];
%                 ss4curves = [0.1,0.7,0.9];
                
                so=STDs(1);
                mxsd=rmax;
                a1=0;
                a2=0;
                rz(1) = 1e3; %step (1/rz): std ratio
                rz(2) = 1e3; %step (1/rz): correlação
                rz(3) = 1e3; %fator de arredondamento
                
                std_rt = 1/rz(2):1/rz(2):2;
                correl = -1:1/rz(1):1;
                
                for sr = std_rt
                    a1=a1+1;
                    for crr = correl
                        %                     sr = sm/so;
                        a2=a2+1;
                        SS4(a1,a2) = ((1+crr)^4)/(4*(sr+1/sr)^2);
                        SS4(a1,a2) = round(SS4(a1,a2)*rz(3))/rz(3);
                    end
                    a2=0;
                end; clear crr sr
                
                
                for i = 1:length(ss4curves)
                    [isr{i},icr{i}]=find(SS4==ss4curves(i));
                    ang2 = linspace(0,2*pi,length(isr{i}));
                    
                    crr{i} = correl(icr{i});
                    sr{i} = std_rt(isr{i});
                    
                    [SS4std, idx] = sort(sr{i}*so,'ascend');
                    SS4crr = crr{i}(idx);
                    
%                     rms{i} = sqrt(repmat(so,1,length(sr{i})).^2 + SS4std.^2 - 2*so*SS4std.*SS4crr);
                    r1 = (SS4std.*cos(real(acos(SS4crr))).^2 + SS4std.*sin(real(acos(SS4crr))).^2).^0.5;
                    r2 = (mxsd*cos(ang2).^2+mxsd*sin(ang2).^2).^0.5;
                    
                    I = find(r1<=r2);
                    xs{i}=smooth(SS4std(I).*cos(real(acos(SS4crr(I)))))';
                    ys{i}=smooth(SS4std(I).*sin(real(acos(SS4crr(I)))))';

                    if min(ys{i})~=0
                        syms E stdr
                        E = 4/((stdr+1/stdr)^2);
                        minSTD = double(solve([E == num2str(ss4curves(i))]))*so;
                        [~,imin] = min(abs(minSTD-min(xs{i})));
                        minSTD = minSTD(imin);

                        xs{i} = [minSTD, xs{i}];
                        ys{i} = [0, ys{i}];
                    end
                    
                    plot(xs{i},ys{i},'--r','linewidth',1);
                    
                    tang = 175;
                    y=@(x)tand(tang)*(x-so);
                    x=linspace(0,rmax,length(xs{i}));
                    Int = InterX([xs{i}; ys{i}],[x; y(x)]);
                    tickSS4angle = tang-90;
                    
                    text(Int(1,1),Int(2,1), ...
                        [num2str(fix(ss4curves(i)*100)/100)],'verticalalignment','bottom',...
                        'handlevisibility','off','parent',cax,'color','r','rotation',tickSS4angle,'fontsize',CFZ.rmsd,'fontname',fNM)
                end
                
                tang = 160;
                y=@(x)tand(tang)*(x-so);
                x=linspace(0,rmax,length(xs{1}));
                Int = InterX([xs{ss4curves==0.5}; ys{ss4curves==0.5}],[x; y(x)]);
                
                tickSS4angle = tang-85;
                
                text(Int(1,1),Int(2,1), ...
                    ['SS4'],'verticalalignment','bottom',...
                    'handlevisibility','off','parent',cax,'color','r', ...
                    'rotation',tickSS4angle,'fontweight','bold', ...
                    'fontsize',CFZ.rmsd,'fontname',fNM);
                
            end
            
            if plotRMS
                % ANGLE OF THE TICK LABELS
                c82 = cos(tickRMSangle*pi/180);
                s82 = sin(tickRMSangle*pi/180);
                for ic = 1 : length(tickRMS)
                    i = tickRMS(ic);
                    iphic = find( sqrt(dx^2+rmax^2-2*dx*rmax*xunit) >= i ,1);
                    ig = find(i*cos(th)+dx <= rmax*cos(th(iphic)));
                    hhh = line(xunit(ig)*i+dx,yunit(ig)*i,'linestyle',styleRMS,'color','k','linewidth',widthRMS,...
                        'handlevisibility','off','parent',cax);
                    if showlabelsRMS
                        text((i+rincRMS/20)*c82+dx,(i+rincRMS/20)*s82, ...
                            ['  ' num2str(fix(i*100)/100)],'verticalalignment','bottom',...
                            'handlevisibility','off','parent',cax,'color','k','rotation',tickRMSangle-90,'fontsize',CFZ.rmsd,'fontname',fNM)
                    end
                end
            end
            % DRAW CORRELATIONS LINES EMANATING FROM THE ORIGIN:
            Corr = tickCOR(Npan).val;
            %             corr = fliplr([0:0.05:1]);
            th  = acos(Corr);
            cst = cos(th); snt = sin(th);
            cs = [-cst; cst];
            sn = [-snt; snt];
            rmxcs = rmax*cs;
            rmxsn = rmax*sn;
            for i = 1:length(rmxcs)
                line(rmxcs(:,i),rmxsn(:,i),'linestyle',styleCOR,'color','k','linewidth',widthCOR,...
                    'handlevisibility','off','parent',cax)
                if i == 8 % cor = 0.5
                    line(rmxcs(:,i),rmxsn(:,i),'linestyle','-','color','k','linewidth',widthCOR,...
                        'handlevisibility','off','parent',cax);
                    pkl4(1,:) = rmxcs(:,i);
                    pkl4(2,:) = rmxsn(:,i);
                end
            end
            
            % DRAW DIFFERENTLY THE CIRCLE CORRESPONDING TO THE OBSERVED VALUE
            %      hhh = line((cos(th)*dx),sin(th)*dx,'linestyle','--','color',colSTD,'linewidth',1,...
            %                   'handlevisibility','off','parent',cax);
            tickSTD2 = tickSTD;
            tickSTD2(end+3) = tickSTD2(end);
            tickSTD2(end-3) = STDs(1);
            tickSTD2(end-2) = STDs(1)*0.75;
            tickSTD2(end-1) = STDs(1)*1.25;
            
            % DRAW STD CIRCLES:
            % draw radial circles
            for ic = 1 : length(tickSTD2)
                i = tickSTD2(ic);
                xuniti = fix(xunit*i*10^6)/10^6;
                yuniti = fix(yunit*i*10^6)/10^6;
                if ic == length(tickSTD2)-3
                    hhh = line(xuniti,yuniti,'linestyle','-','color','m','linewidth',widthSTD,...
                        'handlevisibility','off','parent',cax);
                elseif ic == length(tickSTD2)-2 || ic == length(tickSTD2)-1
                    
                    L1(1,:) = xuniti; L1(2,:) = yuniti;
                    L2(1,:) = rmxcs(:,8); L2(2,:) = rmxsn(:,8);
                    
                    P = InterX(L1,L2);
                    [~,limmax] = min(abs(P(1,P(1,:)>0)-xuniti));
                    hhh = line(xuniti(1:limmax),yuniti(1:limmax),'linestyle','-','color','k','linewidth',widthSTD,...
                        'handlevisibility','off','parent',cax);
                    pkl3(1,ic,:) = xuniti(1:limmax);
                    pkl3(2,ic,:) = yuniti(1:limmax);
                    %             fill(xuniti(1:limmax),yuniti(1:limmax),'r');
                elseif ic == length(tickSTD2)
                    hhh = line(xuniti,yuniti,'linestyle',styleSTD,'color',colSTD,'linewidth',widthSTD,...
                        'handlevisibility','off','parent',cax);
                    Xfill_end = xuniti;
                    Yfill_end = yuniti;
                    FILL(2) = fill(Xfill_end,Yfill_end,'y');
                    alpha(.1)
                else
                    hhh = line(xuniti,yuniti,'linestyle',styleSTD,'color',colSTD,'linewidth',widthSTD,...
                        'handlevisibility','off','parent',cax);
                end
                if showlabelsSTD
                    if Npan == 2
                        if length(find(tickSTD2==0)) == 0
                            text(0,-rinc/20,'0','verticalalignment','top','horizontalAlignment','center',...
                                'handlevisibility','off','parent',cax,'color',colSTD,'fontname',fNM);
                        end
                        text(i,-rinc/20, ...
                            num2str(i),'verticalalignment','top','horizontalAlignment','center',...
                            'handlevisibility','off','parent',cax,'color',colSTD,'fontname',fNM)
                    else
                        if length(find(tickSTD2==0)) == 0
                            %                             text(-rinc/20,rinc/20,'0','verticalalignment','middle','horizontalAlignment','right',...
                            %                                 'handlevisibility','off','parent',cax,'color',colSTD,'fontsize',CFZ.std);
                            text(-rinc/20,rinc/20,'0','verticalalignment',POS.std,'horizontalAlignment','right',...
                                'handlevisibility','off','parent',cax,'color',colSTD,'fontsize',CFZ.std,'fontname',fNM);
                            
                        end
                        if ic ~= length(tickSTD2)-2 && ic ~= length(tickSTD2)-1
                            %                             text(-rinc/20,i, ...
                            %                                 num2str(fix(i*100)/100),'verticalalignment','middle','horizontalAlignment','right',...
                            %                                 'handlevisibility','off','parent',cax,'color',colSTD,'fontsize',CFZ.std)
                            text(-rinc/20,i, ...
                                num2str(fix(i*100)/100),'verticalalignment',POS.std,'horizontalAlignment','right',...
                                'handlevisibility','off','parent',cax,'color',colSTD,'fontsize',CFZ.std,'fontname',fNM)
                        end
                    end
                end
            end
            set(hhh,'linestyle','-') % Make outer circle solid
            Xfill = [permute(pkl3(1,size(pkl3,2)-1,:),[3 2 1])' flip(permute(pkl3(1,size(pkl3,2),:),[3 2 1]))'];
            Yfill = [permute(pkl3(2,size(pkl3,2)-1,:),[3 2 1])' flip(permute(pkl3(2,size(pkl3,2),:),[3 2 1]))'];
            %     rgb_roxo = [128 0 128]/255;
            FILL(1)=fill(Xfill,Yfill,'b');
            alpha(.1)
            
            Xfill = [pkl4(1,2) Xfill_end(Xfill_end>=0 & Xfill_end<=pkl4(1,2))];
            Yfill = [pkl4(2,2) Yfill_end(Yfill_end>=pkl4(2,2))];
            [~,mx_f] = max(Yfill);
            Xfill = flip(Xfill(1:mx_f));
            Yfill = flip(Yfill(1:mx_f));
            Xfill(end+1) = 0;
            Yfill(end+1) = 0;
            FILL(3) = fill(Xfill,Yfill,'m');
            alpha(.1)
            
            % annotate them in correlation coef
            if showlabelsCOR
                rt = 1.05*rmax;
                for i = 1:length(Corr)
                    text(rt*cst(i),rt*snt(i),num2str(Corr(i)),...
                        'horizontalalignment','center',...
                        'handlevisibility','off','parent',cax,'color','k', ...
                        'fontsize',CFZ.corr,'fontname',fNM);
                    %                     text(rt*cst(i),rt*snt(i),num2str(corr(i)),...
                    %                         'horizontalalignment','center',...
                    %                         'handlevisibility','off','parent',cax,'color','k');
                    if i == length(Corr)
                        loc = int2str(0);
                        loc = '1';
                    else
                        loc = int2str(180+i*30);
                        loc = '-1';
                    end
                end
            end
            
            % AXIS TITLES
            axlabweight = 'bold';
            ix = 0;
            if Npan == 1
                if titleSTD
                    ix = ix + 1;
                    ax(ix).handle = ylabel('Standard deviation (MW)','color',colSTD,'fontweight',axlabweight,'fontsize',LAB.std,'fontname',fNM);
                    STDLAB = ax(ix).handle;
                    %                     set(ax(ix).handle, 'position', get(ax(ix).handle,'position')-[0.01,0,0]);
                end
                
                if titleCOR
                    ix = ix + 1;
                    clear ttt
                    pos1 = 45;	DA = 15;
                    lab = 'Correlation Coefficient';
                    c = fliplr(linspace(pos1-DA,pos1+DA,length(lab)));
                    dd = 1.1*rmax;	ii = 0;
                    for ic = 1 : length(c)
                        ith = c(ic);
                        ii = ii + 1;
                        ttt(ii)=text(dd*cos(ith*pi/180),dd*sin(ith*pi/180),lab(ii));
                        set(ttt(ii),'rotation',ith-90,'color','k','horizontalalignment','center',...
                            'verticalalignment','baseline','fontsize',LAB.corr,'fontweight',axlabweight,'fontname',fNM);
                    end
                    ax(ix).handle = ttt;
                end
                
                %                 if titleRMS
                %                     ix = ix + 1;
                %                     clear ttt
                %                     pos1 = tickRMSangle+(180-tickRMSangle)/2; DA = 15; pos1 = 160;
                %                     lab = 'RMSD';
                %                     c = fliplr(linspace(pos1-DA,pos1+DA,length(lab)));
                %                     dd = 1.05*tickRMS(1);
                %                     dd = .95*tickRMS(2);
                %                     ii = 0;
                %                     for ic = 1 : length(c)
                %                         ith = c(ic);
                %                         ii = ii + 1;
                %                         ttt(ii)=text(dx+dd*cos(ith*pi/180),dd*sin(ith*pi/180),lab(ii));
                %                         set(ttt(ii),'rotation',ith-90,'color','k','horizontalalignment','center',...
                %                             'verticalalignment','bottom','fontsize',get(ax(1).handle,'fontsize'),'fontweight',axlabweight,'fontname',fNM);
                %                     end
                %                     ax(ix).handle = ttt;
                %                 end
                
                if titleRMS && plotRMS
                    ix = ix + 1;
                    clear ttt
                    pos1 = tickRMSangle+(180-tickRMSangle)/2; DA = 15; pos1 = 160;
                    lab = 'RMSD (MW)';
                    c = fliplr(linspace(pos1-DA,pos1+DA,length(lab)));
                    dd = 1.05*tickRMS(1);
                    dd = .95*tickRMS(2);
                    ii = 0;
                    for ic = 1 : length(c)
                        ith = c(ic);
                        ii = ii + 1;
                        ttt(ii)=text(dx+dd*cos(ith*pi/180),dd*sin(ith*pi/180),lab(ii));
                        set(ttt(ii),'rotation',ith-90,'color','k','horizontalalignment','center',...
                            'verticalalignment','bottom','fontsize',get(ax(1).handle,'fontsize'),'fontweight',axlabweight,'fontname',fNM);
                    end
                    ax(ix).handle = ttt;
                end
                
            else
                if titleSTD
                    ix = ix + 1;
                    ax(ix).handle =xlabel('Standard deviation','fontweight',axlabweight,'color',colSTD);
                    %             set(ax(ix).handle, 'position', get(ax(ix).handle,'position')-[0.1,0,0]);
                end
                
                if titleCOR
                    ix = ix + 1;
                    clear ttt
                    pos1 = 90;	DA = 15;
                    lab = 'Correlation Coefficient';
                    c = fliplr(linspace(pos1-DA,pos1+DA,length(lab)));
                    dd = 1.1*rmax;	ii = 0;
                    for ic = 1 : length(c)
                        ith = c(ic);
                        ii = ii + 1;
                        ttt(ii)=text(dd*cos(ith*pi/180),dd*sin(ith*pi/180),lab(ii));
                        set(ttt(ii),'rotation',ith-90,'color',colCOR,'horizontalalignment','center',...
                            'verticalalignment','bottom','fontsize',get(ax(1).handle,'fontsize'),'fontweight',axlabweight,'fontname',fNM);
                    end
                    ax(ix).handle = ttt;
                end
                
                if titleRMS
                    ix = ix + 1;
                    clear ttt
                    pos1 = 160; DA = 10;
                    lab = 'RMSD';
                    c = fliplr(linspace(pos1-DA,pos1+DA,length(lab)));
                    dd = 1.05*tickRMS(1); ii = 0;
                    for ic = 1 : length(c)
                        ith = c(ic);
                        ii = ii + 1;
                        ttt(ii)=text(dx+dd*cos(ith*pi/180),dd*sin(ith*pi/180),lab(ii));
                        set(ttt(ii),'rotation',ith-90,'color',colRMS,'horizontalalignment','center',...
                            'verticalalignment','bottom','fontsize',get(ax(1).handle,'fontsize'),'fontweight',axlabweight,'fontname',fNM);
                    end
                    ax(ix).handle = ttt;
                end
            end
            
            
            % VARIOUS ADJUSTMENTS TO THE PLOT:
            set(cax,'dataaspectratio',[1 1 1]), axis(cax,'off'); set(cax,'NextPlot',next);
            set(get(cax,'xlabel'),'visible','on')
            set(get(cax,'ylabel'),'visible','on')
            %    makemcode('RegisterHandle',cax,'IgnoreHandle',q,'FunctionName','polar');
            % set view to 2-D
            view(cax,2);
            % set axis limits
            if Npan == 2
                axis(cax,rmax*[-1.15 1.15 0 1.15]);
                line([-rmax rmax],[0 0],'color',tc,'linewidth',1.2);
                line([0 0],[0 rmax],'color',tc);
            else
                axis(cax,rmax*[-0.035 1.15 0 1.15]);
                %	    axis(cax,rmax*[-1 1 -1.15 1.15]);
                line([0 rmax],[0 0],'color',tc,'linewidth',1.2);
                line([0 0],[0 rmax],'color',tc,'linewidth',2);
            end
            
            %             set(STDLAB, 'position', get(STDLAB,'position')+[0.05,0,0]);
            
            
            % FINALY PLOT THE POINTS:
            hold on
            
            j1=1;
            jj=-1;
            
            aux1=(length(STDs)-1)/(size(Pn,2)-1); 
            
            for oo = 1 : length(STDs)
                
                jj=jj+1;
                if oo>2 && round((oo-2)/aux1)==(oo-2)/aux1
                    j1=j1+1;
                    jj=1;
                end
                
                if CORs(oo)~=0
                    
                    %		pp(ii)=polar(theta(ii),rho(ii));
                    if oo==1
                        pp(oo)=plot(rho(oo)*cos(theta(oo)),rho(oo)*sin(theta(oo)),'-','color',colors{oo},'marker','.','markersize',10);
                        tt(oo)=text(rho(oo)*cos(theta(oo)),rho(oo)*sin(theta(oo)),Pn{oo},'color',colors{oo});
                    else
                        pp(oo)=plot(rho(oo)*cos(theta(oo)),rho(oo)*sin(theta(oo)),'.','color',colors{j1+1},'markersize',10);
                        tt(oo)=text(rho(oo)*cos(theta(oo)),rho(oo)*sin(theta(oo)),[Pn{j1+1} '-' num2str(jj)],'color',colors{j1+1});
                    end

                else
                    tt(oo)=text(rho(oo)*cos(theta(oo)),rho(oo)*sin(theta(oo)),'none','color','w');
                end
            end
            set(tt,'verticalalignment','bottom','horizontalalignment','right')
            set(tt,'fontsize',8)
            
            %%% OUTPUT
            switch nargout
                case 1
                    varargout(1) = {pp};
                case 2
                    varargout(1) = {pp};
                    varargout(2) = {tt};
                case 3
                    varargout(1) = {pp};
                    varargout(2) = {tt};
                    varargout(3) = {ax};
                case 4
                    varargout(1) = {pp};
                    varargout(2) = {tt};
                    varargout(3) = {ax};
                    varargout(4) = {FILL};
            end
            
            
            function varargout = disp_optionslist(varargin)
                % Copyright (c) 2010, Guillaume MAZE
                % All rights reserved.
                %
                % Redistribution and use in source and binary forms, with or without
                % modification, are permitted provided that the following conditions are
                % met:
                %
                %     * Redistributions of source code must retain the above copyright
                %       notice, this list of conditions and the following disclaimer.
                %     * Redistributions in binary form must reproduce the above copyright
                %       notice, this list of conditions and the following disclaimer in
                %       the documentation and/or other materials provided with the distribution
                %
                % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
                % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
                % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
                % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
                % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
                % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
                % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
                % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
                % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
                % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
                % POSSIBILITY OF SUCH DAMAGE.
                
                disp('General options:')
                dispopt('''Npan''',sprintf('1 or 2: Panels to display (1 for positive correlations, 2 for positive and negative correlations).\n\t\tDefault value depends on CORs'));
                
                disp('RMS axis options:')
                dispopt('''tickRMS''','RMS values to plot gridding circles from observation point');
                dispopt('''colRMS''','RMS grid and tick labels color. Default: green');
                dispopt('''showlabelsRMS''','0 / 1 (default): Show or not the RMS tick labels');
                dispopt('''tickRMSangle''','Angle for RMS tick lables with the observation point. Default: 135 deg.');
                dispopt('''styleRMS''','Linestyle of the RMS grid');
                dispopt('''widthRMS''','Line width of the RMS grid');
                dispopt('''titleRMS''','0 / 1 (default): Show RMSD axis title');
                
                disp('STD axis options:')
                dispopt('''tickSTD''','STD values to plot gridding circles from origin');
                dispopt('''colSTD''','STD grid and tick labels color. Default: black');
                dispopt('''showlabelsSTD''','0 / 1 (default): Show or not the STD tick labels');
                dispopt('''styleSTD''','Linestyle of the STD grid');
                dispopt('''widthSTD''','Line width of the STD grid');
                dispopt('''titleSTD''','0 / 1 (default): Show STD axis title');
                dispopt('''limSTD''','Max of the STD axis (radius of the largest circle)');
                
                disp('CORRELATION axis options:')
                dispopt('''tickCOR''','CORRELATON grid values');
                dispopt('''colCOR''','CORRELATION grid color. Default: blue');
                dispopt('''showlabelsCOR''','0 / 1 (default): Show or not the CORRELATION tick labels');
                dispopt('''styleCOR''','Linestyle of the COR grid');
                dispopt('''widthCOR''','Line width of the COR grid');
                dispopt('''titleCOR''','0 / 1 (default): Show CORRELATION axis title');
                
            end%function
            function [] = dispopt(optname,optval)
                % Copyright (c) 2010, Guillaume MAZE
                % All rights reserved.
                %
                % Redistribution and use in source and binary forms, with or without
                % modification, are permitted provided that the following conditions are
                % met:
                %
                %     * Redistributions of source code must retain the above copyright
                %       notice, this list of conditions and the following disclaimer.
                %     * Redistributions in binary form must reproduce the above copyright
                %       notice, this list of conditions and the following disclaimer in
                %       the documentation and/or other materials provided with the distribution
                %
                % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
                % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
                % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
                % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
                % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
                % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
                % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
                % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
                % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
                % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
                % POSSIBILITY OF SUCH DAMAGE.
                disp(sprintf('\t%s',optname));
                disp(sprintf('\t\t%s',optval));
            end
            function P = InterX(L1,varargin)
                %INTERX Intersection of curves
                %   P = INTERX(L1,L2) returns the intersection points of two curves L1
                %   and L2. The curves L1,L2 can be either closed or open and are described
                %   by two-row-matrices, where each row contains its x- and y- coordinates.
                %   The intersection of groups of curves (e.g. contour lines, multiply
                %   connected regions etc) can also be computed by separating them with a
                %   column of NaNs as for example
                %
                %         L  = [x11 x12 x13 ... NaN x21 x22 x23 ...;
                %               y11 y12 y13 ... NaN y21 y22 y23 ...]
                %
                %   P has the same structure as L1 and L2, and its rows correspond to the
                %   x- and y- coordinates of the intersection points of L1 and L2. If no
                %   intersections are found, the returned P is empty.
                %
                %   P = INTERX(L1) returns the self-intersection points of L1. To keep
                %   the code simple, the points at which the curve is tangent to itself are
                %   not included. P = INTERX(L1,L1) returns all the points of the curve
                %   together with any self-intersection points.
                %
                %   Example:
                %       t = linspace(0,2*pi);
                %       r1 = sin(4*t)+2;  x1 = r1.*cos(t); y1 = r1.*sin(t);
                %       r2 = sin(8*t)+2;  x2 = r2.*cos(t); y2 = r2.*sin(t);
                %       P = InterX([x1;y1],[x2;y2]);
                %       plot(x1,y1,x2,y2,P(1,:),P(2,:),'ro')
                
                %   Author : NS
                %   Version: 3.0, 21 Sept. 2010
                
                %   Two words about the algorithm: Most of the code is self-explanatory.
                %   The only trick lies in the calculation of C1 and C2. To be brief, this
                %   is essentially the two-dimensional analog of the condition that needs
                %   to be satisfied by a function F(x) that has a zero in the interval
                %   [a,b], namely
                %           F(a)*F(b) <= 0
                %   C1 and C2 exactly do this for each segment of curves 1 and 2
                %   respectively. If this condition is satisfied simultaneously for two
                %   segments then we know that they will cross at some point.
                %   Each factor of the 'C' arrays is essentially a matrix containing
                %   the numerators of the signed distances between points of one curve
                %   and line segments of the other.
                
                %...Argument checks and assignment of L2
                %             error(nargchk(1,2,nargin));
                if nargin == 1,
                    LL2 = L1;    hF = @lt;   %...Avoid the inclusion of common points
                else
                    LL2 = varargin{1}; hF = @le;
                end
                
                %...Preliminary stuff
                x1  = L1(1,:)';  x2 = LL2(1,:);
                y1  = L1(2,:)';  y2 = LL2(2,:);
                dx1 = diff(x1); dy1 = diff(y1);
                dx2 = diff(x2); dy2 = diff(y2);
                
                %...Determine 'signed distances'
                S1 = dx1.*y1(1:end-1) - dy1.*x1(1:end-1);
                S2 = dx2.*y2(1:end-1) - dy2.*x2(1:end-1);
                
                C1 = feval(hF,D(bsxfun(@times,dx1,y2)-bsxfun(@times,dy1,x2),S1),0);
                C2 = feval(hF,D((bsxfun(@times,y1,dx2)-bsxfun(@times,x1,dy2))',S2'),0)';
                
                %...Obtain the segments where an intersection is expected
                [iy,j] = find(C1 & C2);
                if isempty(iy),P = zeros(2,0);return; end;
                
                %...Transpose and prepare for output
                iy=iy'; dx2=dx2'; dy2=dy2'; S2 = S2';
                L = dy2(j).*dx1(iy) - dy1(iy).*dx2(j);
                iy = iy(L~=0); j=j(L~=0); L=L(L~=0);  %...Avoid divisions by 0
                
                %...Solve system of eqs to get the common points
                P = unique([dx2(j).*S1(iy) - dx1(iy).*S2(j), ...
                    dy2(j).*S1(iy) - dy1(iy).*S2(j)]./[L L],'rows')';
                
                function u = D(x,y)
                    u = bsxfun(@minus,x(:,1:end-1),y).*bsxfun(@minus,x(:,2:end),y);
                end
            end
        end%function