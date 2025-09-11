#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on September 11, 2025, at 14:43
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'squares'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'Gender': ["Male", "Female", "Non-binary", "Prefer not to say"],
    'Age': [18,19,20,21,22,23,24,'26+'],
    'I consent*': False,
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Michael\\OneDrive - Georgia Southern University\\4_RESEARCH\\behavior_contrast\\behavioral_contrast_psychopy\\main_csv_2_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=2,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "main_square_exp_2" ---
    logging_mouse = event.Mouse(win=win)
    x, y = [None, None]
    logging_mouse.mouseClock = core.Clock()
    click_square = visual.Rect(
        win=win, name='click_square',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    mouse_score = event.Mouse(win=win)
    x, y = [None, None]
    mouse_score.mouseClock = core.Clock()
    big_text = visual.TextStim(win=win, name='big_text',
        text='',
        font='Arial',
        pos=(-.5, 0.1), draggable=False, height=0.06, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from code
    score = 0
    button_2 = visual.ButtonStim(win, 
        text='Reset Global Clock', font='Arvo',
        pos=(0.5, 0.5),
        letterHeight=0.025,
        size=(0.3, 0.3), 
        ori=0.0
        ,borderWidth=2.0,
        fillColor='black', borderColor='red',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_2',
        depth=-5
    )
    button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "points_square_exp_2" ---
    mouse_logging_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_logging_2.mouseClock = core.Clock()
    button = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(0.4, -0.4),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    score_textbox_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, 0.4), draggable=False,      letterHeight=0.1,
         size=(0.2, 0.2), borderWidth=0.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='score_textbox_2',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "routine_1" ---
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    Instructions_phase = visual.TextStim(win=win, name='Instructions_phase',
        text=' ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    outer_trials = data.TrialHandler2(
        name='outer_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('bin/.csv/short_loop.csv'), 
        seed=None, 
    )
    thisExp.addLoop(outer_trials)  # add the loop to the experiment
    thisOuter_trial = outer_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_trial.rgb)
    if thisOuter_trial != None:
        for paramName in thisOuter_trial:
            globals()[paramName] = thisOuter_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisOuter_trial in outer_trials:
        outer_trials.status = STARTED
        if hasattr(thisOuter_trial, 'status'):
            thisOuter_trial.status = STARTED
        currentLoop = outer_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisOuter_trial.rgb)
        if thisOuter_trial != None:
            for paramName in thisOuter_trial:
                globals()[paramName] = thisOuter_trial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        inner_trials = data.TrialHandler2(
            name='inner_trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(compnt_file), 
            seed=None, 
        )
        thisExp.addLoop(inner_trials)  # add the loop to the experiment
        thisInner_trial = inner_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInner_trial.rgb)
        if thisInner_trial != None:
            for paramName in thisInner_trial:
                globals()[paramName] = thisInner_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisInner_trial in inner_trials:
            inner_trials.status = STARTED
            if hasattr(thisInner_trial, 'status'):
                thisInner_trial.status = STARTED
            currentLoop = inner_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisInner_trial.rgb)
            if thisInner_trial != None:
                for paramName in thisInner_trial:
                    globals()[paramName] = thisInner_trial[paramName]
            
            # --- Prepare to start Routine "main_square_exp_2" ---
            # create an object to store info about Routine main_square_exp_2
            main_square_exp_2 = data.Routine(
                name='main_square_exp_2',
                components=[logging_mouse, click_square, mouse_score, big_text, button_2],
            )
            main_square_exp_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the logging_mouse
            logging_mouse.x = []
            logging_mouse.y = []
            logging_mouse.leftButton = []
            logging_mouse.midButton = []
            logging_mouse.rightButton = []
            logging_mouse.time = []
            gotValidClick = False  # until a click is received
            # setup some python lists for storing info about the mouse_score
            mouse_score.x = []
            mouse_score.y = []
            mouse_score.leftButton = []
            mouse_score.midButton = []
            mouse_score.rightButton = []
            mouse_score.time = []
            mouse_score.corr = []
            mouse_score.clicked_name = []
            gotValidClick = False  # until a click is received
            # reset button_2 to account for continued clicks & clear times on/off
            button_2.reset()
            # store start times for main_square_exp_2
            main_square_exp_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            main_square_exp_2.tStart = globalClock.getTime(format='float')
            main_square_exp_2.status = STARTED
            thisExp.addData('main_square_exp_2.started', main_square_exp_2.tStart)
            main_square_exp_2.maxDuration = None
            # skip Routine main_square_exp_2 if its 'Skip if' condition is True
            main_square_exp_2.skipped = continueRoutine and not (globalClock.getTime() > end_time)
            continueRoutine = main_square_exp_2.skipped
            win.color = background
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            main_square_exp_2Components = main_square_exp_2.components
            for thisComponent in main_square_exp_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "main_square_exp_2" ---
            main_square_exp_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial, 'status') and thisInner_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # *logging_mouse* updates
                
                # if logging_mouse is starting this frame...
                if logging_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    logging_mouse.frameNStart = frameN  # exact frame index
                    logging_mouse.tStart = t  # local t and not account for scr refresh
                    logging_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(logging_mouse, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    logging_mouse.status = STARTED
                    prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
                
                # if logging_mouse is stopping this frame...
                if logging_mouse.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > logging_mouse.tStartRefresh + scoring_interval-frameTolerance:
                        # keep track of stop time/frame for later
                        logging_mouse.tStop = t  # not accounting for scr refresh
                        logging_mouse.tStopRefresh = tThisFlipGlobal  # on global time
                        logging_mouse.frameNStop = frameN  # exact frame index
                        # update status
                        logging_mouse.status = FINISHED
                if logging_mouse.status == STARTED:  # only update if started and not finished!
                    buttons = logging_mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = logging_mouse.getPos()
                            logging_mouse.x.append(x)
                            logging_mouse.y.append(y)
                            buttons = logging_mouse.getPressed()
                            logging_mouse.leftButton.append(buttons[0])
                            logging_mouse.midButton.append(buttons[1])
                            logging_mouse.rightButton.append(buttons[2])
                            logging_mouse.time.append(globalClock.getTime())
                
                # *click_square* updates
                
                # if click_square is starting this frame...
                if click_square.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    click_square.frameNStart = frameN  # exact frame index
                    click_square.tStart = t  # local t and not account for scr refresh
                    click_square.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(click_square, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'click_square.started')
                    # update status
                    click_square.status = STARTED
                    click_square.setAutoDraw(True)
                
                # if click_square is active this frame...
                if click_square.status == STARTED:
                    # update params
                    pass
                # *mouse_score* updates
                
                # if mouse_score is starting this frame...
                if mouse_score.status == NOT_STARTED and t >= scoring_interval-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_score.frameNStart = frameN  # exact frame index
                    mouse_score.tStart = t  # local t and not account for scr refresh
                    mouse_score.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_score, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse_score.started', t)
                    # update status
                    mouse_score.status = STARTED
                    prevButtonState = mouse_score.getPressed()  # if button is down already this ISN'T a new click
                if mouse_score.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_score.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(click_square, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse_score):
                                    gotValidClick = True
                                    mouse_score.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse_score.clicked_name.append(None)
                            # check whether click was in correct object
                            if gotValidClick:
                                _corr = 0
                                _corrAns = environmenttools.getFromNames(click_square, namespace=locals())
                                for obj in _corrAns:
                                    # is this object clicked on?
                                    if obj.contains(mouse_score):
                                        _corr = 1
                                mouse_score.corr.append(_corr)
                            x, y = mouse_score.getPos()
                            mouse_score.x.append(x)
                            mouse_score.y.append(y)
                            buttons = mouse_score.getPressed()
                            mouse_score.leftButton.append(buttons[0])
                            mouse_score.midButton.append(buttons[1])
                            mouse_score.rightButton.append(buttons[2])
                            mouse_score.time.append(globalClock.getTime())
                            if gotValidClick:
                                continueRoutine = False  # end routine on response
                
                # *big_text* updates
                
                # if big_text is starting this frame...
                if big_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    big_text.frameNStart = frameN  # exact frame index
                    big_text.tStart = t  # local t and not account for scr refresh
                    big_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(big_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'big_text.started')
                    # update status
                    big_text.status = STARTED
                    big_text.setAutoDraw(True)
                
                # if big_text is active this frame...
                if big_text.status == STARTED:
                    # update params
                    big_text.setText(f'''
                    score:{score}
                    score interval:{scoring_interval}
                    clock {int(t)}
                    Global Clock: {int(globalClock.getTime())}
                    Core Clock:{int(core.getTime())}
                    ''', log=False)
                # Run 'Each Frame' code from code
                if globalClock.getTime() > end_time:
                    print(inner_trials.nRemaining)
                    skip = int(inner_trials.nRemaining)
                    inner_trials.skipTrials(skip)
                # *button_2* updates
                
                # if button_2 is starting this frame...
                if button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_2.frameNStart = frameN  # exact frame index
                    button_2.tStart = t  # local t and not account for scr refresh
                    button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button_2.started')
                    # update status
                    button_2.status = STARTED
                    win.callOnFlip(button_2.buttonClock.reset)
                    button_2.setAutoDraw(True)
                
                # if button_2 is active this frame...
                if button_2.status == STARTED:
                    # update params
                    pass
                    # check whether button_2 has been pressed
                    if button_2.isClicked:
                        if not button_2.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_2.timesOn.append(button_2.buttonClock.getTime())
                            button_2.timesOff.append(button_2.buttonClock.getTime())
                        elif len(button_2.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_2.timesOff[-1] = button_2.buttonClock.getTime()
                        if not button_2.wasClicked:
                            # run callback code when button_2 is clicked
                            globalClock.reset()
                # take note of whether button_2 was clicked, so that next frame we know if clicks are new
                button_2.wasClicked = button_2.isClicked and button_2.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=main_square_exp_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    main_square_exp_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in main_square_exp_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "main_square_exp_2" ---
            for thisComponent in main_square_exp_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for main_square_exp_2
            main_square_exp_2.tStop = globalClock.getTime(format='float')
            main_square_exp_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('main_square_exp_2.stopped', main_square_exp_2.tStop)
            setupWindow(expInfo=expInfo, win=win)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('logging_mouse.x', logging_mouse.x)
            inner_trials.addData('logging_mouse.y', logging_mouse.y)
            inner_trials.addData('logging_mouse.leftButton', logging_mouse.leftButton)
            inner_trials.addData('logging_mouse.midButton', logging_mouse.midButton)
            inner_trials.addData('logging_mouse.rightButton', logging_mouse.rightButton)
            inner_trials.addData('logging_mouse.time', logging_mouse.time)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('mouse_score.x', mouse_score.x)
            inner_trials.addData('mouse_score.y', mouse_score.y)
            inner_trials.addData('mouse_score.leftButton', mouse_score.leftButton)
            inner_trials.addData('mouse_score.midButton', mouse_score.midButton)
            inner_trials.addData('mouse_score.rightButton', mouse_score.rightButton)
            inner_trials.addData('mouse_score.time', mouse_score.time)
            inner_trials.addData('mouse_score.corr', mouse_score.corr)
            inner_trials.addData('mouse_score.clicked_name', mouse_score.clicked_name)
            inner_trials.addData('button_2.numClicks', button_2.numClicks)
            if button_2.numClicks:
               inner_trials.addData('button_2.timesOn', button_2.timesOn)
               inner_trials.addData('button_2.timesOff', button_2.timesOff)
            else:
               inner_trials.addData('button_2.timesOn', "")
               inner_trials.addData('button_2.timesOff', "")
            # the Routine "main_square_exp_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "points_square_exp_2" ---
            # create an object to store info about Routine points_square_exp_2
            points_square_exp_2 = data.Routine(
                name='points_square_exp_2',
                components=[mouse_logging_2, button, score_textbox_2],
            )
            points_square_exp_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the mouse_logging_2
            mouse_logging_2.x = []
            mouse_logging_2.y = []
            mouse_logging_2.leftButton = []
            mouse_logging_2.midButton = []
            mouse_logging_2.rightButton = []
            mouse_logging_2.time = []
            gotValidClick = False  # until a click is received
            # reset button to account for continued clicks & clear times on/off
            button.reset()
            score_textbox_2.reset()
            score_textbox_2.setText(score)
            # store start times for points_square_exp_2
            points_square_exp_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            points_square_exp_2.tStart = globalClock.getTime(format='float')
            points_square_exp_2.status = STARTED
            thisExp.addData('points_square_exp_2.started', points_square_exp_2.tStart)
            points_square_exp_2.maxDuration = 10
            # skip Routine points_square_exp_2 if its 'Skip if' condition is True
            points_square_exp_2.skipped = continueRoutine and not (globalClock.getTime() > end_time)
            continueRoutine = points_square_exp_2.skipped
            win.color = background
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            points_square_exp_2Components = points_square_exp_2.components
            for thisComponent in points_square_exp_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "points_square_exp_2" ---
            points_square_exp_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial, 'status') and thisInner_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > points_square_exp_2.maxDuration-frameTolerance:
                    points_square_exp_2.maxDurationReached = True
                    continueRoutine = False
                # *mouse_logging_2* updates
                
                # if mouse_logging_2 is starting this frame...
                if mouse_logging_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_logging_2.frameNStart = frameN  # exact frame index
                    mouse_logging_2.tStart = t  # local t and not account for scr refresh
                    mouse_logging_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_logging_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse_logging_2.started', t)
                    # update status
                    mouse_logging_2.status = STARTED
                    prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
                if mouse_logging_2.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_logging_2.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = mouse_logging_2.getPos()
                            mouse_logging_2.x.append(x)
                            mouse_logging_2.y.append(y)
                            buttons = mouse_logging_2.getPressed()
                            mouse_logging_2.leftButton.append(buttons[0])
                            mouse_logging_2.midButton.append(buttons[1])
                            mouse_logging_2.rightButton.append(buttons[2])
                            mouse_logging_2.time.append(globalClock.getTime())
                # *button* updates
                
                # if button is starting this frame...
                if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button.frameNStart = frameN  # exact frame index
                    button.tStart = t  # local t and not account for scr refresh
                    button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    button.status = STARTED
                    win.callOnFlip(button.buttonClock.reset)
                    button.setAutoDraw(True)
                
                # if button is active this frame...
                if button.status == STARTED:
                    # update params
                    pass
                    # check whether button has been pressed
                    if button.isClicked:
                        if not button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button.timesOn.append(button.buttonClock.getTime())
                            button.timesOff.append(button.buttonClock.getTime())
                        elif len(button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button.timesOff[-1] = button.buttonClock.getTime()
                        if not button.wasClicked:
                            # end routine when button is clicked
                            continueRoutine = False
                        if not button.wasClicked:
                            # run callback code when button is clicked
                            score += points
                # take note of whether button was clicked, so that next frame we know if clicks are new
                button.wasClicked = button.isClicked and button.status == STARTED
                
                # *score_textbox_2* updates
                
                # if score_textbox_2 is starting this frame...
                if score_textbox_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    score_textbox_2.frameNStart = frameN  # exact frame index
                    score_textbox_2.tStart = t  # local t and not account for scr refresh
                    score_textbox_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(score_textbox_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    score_textbox_2.status = STARTED
                    score_textbox_2.setAutoDraw(True)
                
                # if score_textbox_2 is active this frame...
                if score_textbox_2.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=points_square_exp_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    points_square_exp_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in points_square_exp_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "points_square_exp_2" ---
            for thisComponent in points_square_exp_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for points_square_exp_2
            points_square_exp_2.tStop = globalClock.getTime(format='float')
            points_square_exp_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('points_square_exp_2.stopped', points_square_exp_2.tStop)
            setupWindow(expInfo=expInfo, win=win)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('mouse_logging_2.x', mouse_logging_2.x)
            inner_trials.addData('mouse_logging_2.y', mouse_logging_2.y)
            inner_trials.addData('mouse_logging_2.leftButton', mouse_logging_2.leftButton)
            inner_trials.addData('mouse_logging_2.midButton', mouse_logging_2.midButton)
            inner_trials.addData('mouse_logging_2.rightButton', mouse_logging_2.rightButton)
            inner_trials.addData('mouse_logging_2.time', mouse_logging_2.time)
            inner_trials.addData('button.numClicks', button.numClicks)
            if button.numClicks:
               inner_trials.addData('button.timesOn', button.timesOn)
               inner_trials.addData('button.timesOff', button.timesOff)
            else:
               inner_trials.addData('button.timesOn', "")
               inner_trials.addData('button.timesOff', "")
            # the Routine "points_square_exp_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisInner_trial as finished
            if hasattr(thisInner_trial, 'status'):
                thisInner_trial.status = FINISHED
            # if awaiting a pause, pause now
            if inner_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                inner_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'inner_trials'
        inner_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if inner_trials.trialList in ([], [None], None):
            params = []
        else:
            params = inner_trials.trialList[0].keys()
        # save data for this loop
        inner_trials.saveAsText(filename + '_inner_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "routine_1" ---
        # create an object to store info about Routine routine_1
        routine_1 = data.Routine(
            name='routine_1',
            components=[mouse, Instructions_phase],
        )
        routine_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the mouse
        gotValidClick = False  # until a click is received
        # store start times for routine_1
        routine_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        routine_1.tStart = globalClock.getTime(format='float')
        routine_1.status = STARTED
        thisExp.addData('routine_1.started', routine_1.tStart)
        routine_1.maxDuration = 3
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        routine_1Components = routine_1.components
        for thisComponent in routine_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_1" ---
        routine_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisOuter_trial, 'status') and thisOuter_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > routine_1.maxDuration-frameTolerance:
                routine_1.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from code_4
            #win.color =  (0,0,0) #black #background # (1.0000, 1.0000, 1.0000)
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 2-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse.status = STARTED
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        continueRoutine = False  # end routine on response            
            # *Instructions_phase* updates
            
            # if Instructions_phase is starting this frame...
            if Instructions_phase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instructions_phase.frameNStart = frameN  # exact frame index
                Instructions_phase.tStart = t  # local t and not account for scr refresh
                Instructions_phase.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instructions_phase, 'tStartRefresh')  # time at next scr refresh
                # update status
                Instructions_phase.status = STARTED
                Instructions_phase.setAutoDraw(True)
            
            # if Instructions_phase is active this frame...
            if Instructions_phase.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=routine_1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routine_1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_1" ---
        for thisComponent in routine_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for routine_1
        routine_1.tStop = globalClock.getTime(format='float')
        routine_1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('routine_1.stopped', routine_1.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from code_4
        globalClock.reset()  # Resets the global timer to 0
        # stop_routine = False
        # store data for outer_trials (TrialHandler)
        x, y = mouse.getPos()
        buttons = mouse.getPressed()
        outer_trials.addData('mouse.x', x)
        outer_trials.addData('mouse.y', y)
        outer_trials.addData('mouse.leftButton', buttons[0])
        outer_trials.addData('mouse.midButton', buttons[1])
        outer_trials.addData('mouse.rightButton', buttons[2])
        # the Routine "routine_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisOuter_trial as finished
        if hasattr(thisOuter_trial, 'status'):
            thisOuter_trial.status = FINISHED
        # if awaiting a pause, pause now
        if outer_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            outer_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'outer_trials'
    outer_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if outer_trials.trialList in ([], [None], None):
        params = []
    else:
        params = outer_trials.trialList[0].keys()
    # save data for this loop
    outer_trials.saveAsText(filename + '_outer_trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
