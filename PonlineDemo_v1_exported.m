classdef PonlineDemo_v1_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        ConfigurePanel       matlab.ui.container.Panel
        DeviceDropDown       matlab.ui.control.DropDown
        DeviceDropDownLabel  matlab.ui.control.Label
        ConfigureButton      matlab.ui.control.Button
        RecordPanel          matlab.ui.container.Panel
        OfflineButton        matlab.ui.control.Button
        TemplateButton       matlab.ui.control.Button
        UIAxes_grasp         matlab.ui.control.UIAxes
        UIAxes_angle         matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        deviceName 
        probe
        startCommand
        stopCommand
        tcpSocket
        samplingRate
        adn
        gain

        readChannel
        recordChannel
        viewChannel
        readChannelNum
        recordChannelNum
        viewChannelNum
        
        plotEMG
        plotTime
        
        bandpassFilterPara
        combFilterPara

        data_preview
        data_sEMG
        recordFlag
        recordWin

        forceDevice
        forcePara
        mvcData
        mvcPara
        data_force
        plotForce
        plotRefForce
        plotAngle
        plotGrasp
        
        data_labels
        lib_labels

        estimationParameters
        ele_index_all
        stepLen
        munum

        tcp_robot
        refangle
        refforce

        proInd
        
    end
    
    methods (Access = private)
        function data = filterData(app,data)
            for ch = 1:size(data,1)
                if app.BandpassCheckBox.Value
                    data(ch,:) = filter(app.bandpassFilterPara(1,:),app.bandpassFilterPara(2,:),data(ch,:));
                end
                if app.CombCheckBox.Value
                    data(ch,:) = filter(app.combFilterPara(1,:),app.combFilterPara(2,:),data(ch,:));
                end
            end
        end


    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: ConfigureButton
        function ConfigureButtonPushed(app, event)
            app.ConfigureButton.BackgroundColor = [1,1,1];
            app.deviceName = app.DeviceDropDown.Value;
            
            switch app.deviceName
                case 'offline'
                    app.recordChannelNum = length(app.recordChannel);
                    app.adn = 16;
                    app.gain = 150;
                    app.samplingRate = 2048;
            end


            [app.bandpassFilterPara(1,:),app.bandpassFilterPara(2,:)] = butter(4,[20/app.samplingRate*2 500/app.samplingRate*2]);
            fo = 50; % power frequency
            q = 10;
            bw = (fo/(app.samplingRate/2))/q;
            [app.combFilterPara(1,:),app.combFilterPara(2,:)] = iircomb(round(app.samplingRate/fo),bw,'notch');
            app.plotTime = 30;
            fprintf('Configuration completed!\n');
            app.ConfigureButton.BackgroundColor = [0,1,0];
        end


        % Button pushed function: TemplateButton
        function TemplateButtonPushed(app, event)
            fsamp = 2048;
            app.estimationParameters = cell(1,3);
            elenum = 3;
            exFactor = 10;
            % load Mu-tracking templates
            result = load(['./utils/','resultMuCKCTrackingTemplate.mat']);
            decomps = result.result{1,2}{1,1};
            
            for ele = 1:elenum
                ele_index = find(decomps.ele_tag==ele);
                W = decomps.candW(:,ele_index);
                Centroids = decomps.Centroids(ele_index,:);
                app.estimationParameters{ele}.W = W;
                app.estimationParameters{ele}.Centroids = Centroids;
                app.estimationParameters{ele}.exFactor = exFactor;
                app.estimationParameters{ele}.munum = size(W,2);
            end

            app.ele_index_all = arrayfun(@(e) find(decomps.ele_tag == e), 1:elenum, 'UniformOutput', false);
            app.munum = size(decomps.ele_tag,1);
            app.proInd = [6,28 29 64 64+39 64+41 128+16]; 
            app.TemplateButton.BackgroundColor = [0,1,0];

            clear result decomps  

        end

        % Button pushed function: OfflineButton
        function OfflineButtonPushed(app, event)

            app.samplingRate = 2048;
            elenum = 3;
            app.plotAngle = plot(app.UIAxes_angle,0,0,'LineWidth',1,'Color','r');
            app.UIAxes_angle.XLim = [0,app.plotTime];
            app.UIAxes_angle.YLim = [-1,1];
            app.plotGrasp = plot(app.UIAxes_grasp,0,0,'LineWidth',1,'Color','r');
            app.UIAxes_grasp.XLim = [0,app.plotTime];
            app.UIAxes_grasp.YLim = [0,1];
            sub = 15;
            gest = 1;
            dof = 2;
            tr = 2; % 
            fsamp = 2048;
            fs = fsamp;
            % twitch force
            P = 1.2;
            Tr = 80;
            Tl = 10*Tr;
            ttt = 0:1000/fs:Tl;
            twitch = P/Tr.*ttt.*exp(1-ttt./Tr);
            batchsize = 14;
            % load pytorch configurations
            MLP = py.importlib.import_module('module');%加载 angle-LSTM 空模型
            py.importlib.reload(MLP);%加载
            mode = py.importlib.import_module('model_pre');%加载模型参数
            py.importlib.reload(mode);%加载
            models = mode.model_load(pyargs('params_path_a', './utils/lstm_angle.pth','params_path_f', './utils/lstm_force.pth'));
            model_a = models{1};
            model_f = models{2};
            
            predi = py.importlib.import_module('pred');%读取
            py.importlib.reload(predi);%加载
            scaler = py.numpy.load('./utils/scaler.npy');
            scaler = double(scaler);
            pred_array = [];
            pred_array_f = [];
            
            app.OfflineButton.BackgroundColor = [0,1,0];
            app.recordFlag = 1;

            app.recordWin = floor(0.2*app.samplingRate); 
            app.stepLen = round(0.02*app.samplingRate); 
            nCycle_all = inf;
            app.data_sEMG = zeros(app.recordChannelNum,0);
            load(['./offline/',num2str(sub),num2str(gest),num2str(dof),num2str(tr),'.mat']);

     
            batch_size_counter = 0;
            input = zeros(batchsize,6,app.munum);

            nCycle = 0;
            tic;
            while app.recordFlag && nCycle<1400
                nCycle = nCycle+1;
                app.data_sEMG(:,(nCycle-1)*app.stepLen+1:nCycle*app.stepLen) = EMGdata(:,(nCycle-1)*app.stepLen+1:nCycle*app.stepLen);
                if nCycle<10
                    continue
                end
                newEMGdata = app.data_sEMG(:,end-app.recordWin+1:end);
        
                for ele = 1:elenum
                    ele_index = app.ele_index_all{ele};
                    winData = newEMGdata(1+(ele-1)*64:ele*64, :);
                    winData = filter(app.bandpassFilterPara(1,:),app.bandpassFilterPara(2,:), winData, [], 2);
                    winData = filter(app.combFilterPara(1,:),app.combFilterPara(2,:), winData, [], 2);
                    Pulses = winDecomp0406(winData, app.estimationParameters{ele});
                    % twitch-force model
                    for m_index = 1:app.estimationParameters{ele}.munum
                        tmpSpikes = zeros(1, app.recordWin);
                        if ~isempty(Pulses{m_index})
                            tmpSpikes(Pulses{m_index}) = 1;
                        end
                        force_ch = generateForce_ch_0406(tmpSpikes, twitch, Tr);
                        tmpTwitch(ele_index(m_index), nCycle) = mean(force_ch);
                    end
                end

                if nCycle<15
                    continue
                end
                tmp_sample = tmpTwitch(:,end-5:end)';
                batch_size_counter = batch_size_counter+1;
                tmptmp_sample = (tmp_sample-scaler(2))./(scaler(1)-scaler(2));
                input(batch_size_counter,:,:) = tmptmp_sample;

                if batch_size_counter<batchsize
                    continue
                end
                data1 = py.numpy.array(input);
                preds = predi.prediction(pyargs('model_a',model_a,'model_f', model_f,'data', data1));%预测 
                a_pred = double(preds{1});
                f_pred = double(preds{2});
                pred_array = [pred_array; a_pred];
                pred_array_f = [pred_array_f; f_pred];
                start_time = 0;
                if nCycle==28
                    start_time = toc;
                end
                t_angle = start_time+[1:length(pred_array)]/50;
                set(app.plotAngle,'xdata',t_angle,'ydata',pred_array'); % 只更新数据，不更新样式
                set(app.plotGrasp,'xdata',t_angle,'ydata',pred_array_f');
                drawnow limitrate;
                batch_size_counter = 0;
                input = zeros(batchsize,6,app.munum);
     
            end
            app.OfflineButton.BackgroundColor = [1,1,1];
            set(app.plotAngle,'xdata',0,'ydata',0);
            set(app.plotGrasp,'xdata',0,'ydata',0);

        end


    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [1 1 1];
            app.UIFigure.Position = [100 100 1126 511];
            app.UIFigure.Name = 'MATLAB App';

            % Create UIAxes_angle
            app.UIAxes_angle = uiaxes(app.UIFigure);
            title(app.UIAxes_angle, 'Predicted Angle')
            xlabel(app.UIAxes_angle, 'Time (s)')
            ylabel(app.UIAxes_angle, 'Normalized Angle')
            zlabel(app.UIAxes_angle, 'Z')
            app.UIAxes_angle.FontName = 'Times New Roman';
            app.UIAxes_angle.Colormap = [];
            app.UIAxes_angle.YGrid = 'on';
            app.UIAxes_angle.FontSize = 18;
            app.UIAxes_angle.NextPlot = 'add';
            app.UIAxes_angle.Position = [4 21 748 222];

            % Create UIAxes_grasp
            app.UIAxes_grasp = uiaxes(app.UIFigure);
            title(app.UIAxes_grasp, 'Predicted Force')
            xlabel(app.UIAxes_grasp, 'Time (s)')
            ylabel(app.UIAxes_grasp, 'Normalized Force')
            zlabel(app.UIAxes_grasp, 'Z')
            app.UIAxes_grasp.FontName = 'Times New Roman';
            app.UIAxes_grasp.Colormap = zeros(0,3);
            app.UIAxes_grasp.YGrid = 'on';
            app.UIAxes_grasp.FontSize = 18;
            app.UIAxes_grasp.NextPlot = 'add';
            app.UIAxes_grasp.Position = [3 264 749 222];

            % Create RecordPanel
            app.RecordPanel = uipanel(app.UIFigure);
            app.RecordPanel.Title = 'Record';
            app.RecordPanel.BackgroundColor = [1 1 1];
            app.RecordPanel.FontName = 'Times New Roman';
            app.RecordPanel.FontAngle = 'italic';
            app.RecordPanel.FontSize = 18;
            app.RecordPanel.Position = [813 54 301 158];

            % Create TemplateButton
            app.TemplateButton = uibutton(app.RecordPanel, 'push');
            app.TemplateButton.ButtonPushedFcn = createCallbackFcn(app, @TemplateButtonPushed, true);
            app.TemplateButton.BackgroundColor = [1 1 1];
            app.TemplateButton.FontName = 'Times New Roman';
            app.TemplateButton.FontSize = 18;
            app.TemplateButton.Position = [33 47 100 41];
            app.TemplateButton.Text = 'Template';

            % Create OfflineButton
            app.OfflineButton = uibutton(app.RecordPanel, 'push');
            app.OfflineButton.ButtonPushedFcn = createCallbackFcn(app, @OfflineButtonPushed, true);
            app.OfflineButton.BackgroundColor = [1 1 1];
            app.OfflineButton.FontName = 'Times New Roman';
            app.OfflineButton.FontSize = 18;
            app.OfflineButton.Position = [170 47 100 41];
            app.OfflineButton.Text = 'Offline';

            % Create ConfigurePanel
            app.ConfigurePanel = uipanel(app.UIFigure);
            app.ConfigurePanel.Title = 'Configure';
            app.ConfigurePanel.BackgroundColor = [1 1 1];
            app.ConfigurePanel.FontName = 'Times New Roman';
            app.ConfigurePanel.FontAngle = 'italic';
            app.ConfigurePanel.FontSize = 18;
            app.ConfigurePanel.Position = [813 352 302 107];

            % Create ConfigureButton
            app.ConfigureButton = uibutton(app.ConfigurePanel, 'push');
            app.ConfigureButton.ButtonPushedFcn = createCallbackFcn(app, @ConfigureButtonPushed, true);
            app.ConfigureButton.BackgroundColor = [1 1 1];
            app.ConfigureButton.FontName = 'Times New Roman';
            app.ConfigureButton.FontSize = 18;
            app.ConfigureButton.Position = [191 18 100 41];
            app.ConfigureButton.Text = 'Configure';

            % Create DeviceDropDownLabel
            app.DeviceDropDownLabel = uilabel(app.ConfigurePanel);
            app.DeviceDropDownLabel.HorizontalAlignment = 'right';
            app.DeviceDropDownLabel.FontName = 'Times New Roman';
            app.DeviceDropDownLabel.FontSize = 14;
            app.DeviceDropDownLabel.Position = [11 24 45 22];
            app.DeviceDropDownLabel.Text = 'Device';

            % Create DeviceDropDown
            app.DeviceDropDown = uidropdown(app.ConfigurePanel);
            app.DeviceDropDown.Items = {'offline'};
            app.DeviceDropDown.FontName = 'Times New Roman';
            app.DeviceDropDown.FontSize = 14;
            app.DeviceDropDown.Position = [71 24 106 22];
            app.DeviceDropDown.Value = 'offline';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = PonlineDemo_v1_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end