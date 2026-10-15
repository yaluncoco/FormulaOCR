#define MyAppName "FormulaOCR"
#define VersionFileHandle FileOpen(AddBackslash(SourcePath) + "..\VERSION")
#define MyAppVersion Trim(FileRead(VersionFileHandle))
#expr FileClose(VersionFileHandle)
#define MyAppPublisher "yaluncoco"
#define MyAppURL "https://github.com/yaluncoco/FormulaOCR"
#define MyAppExeName "FormulaOCR.exe"
#if GetEnv("FORMULA_OCR_REPO_ROOT") != ""
#define MyRepoRoot GetEnv("FORMULA_OCR_REPO_ROOT")
#else
#define MyRepoRoot SourcePath + ".."
#endif
#define MyBuildDir MyRepoRoot + "\dist\FormulaOCR"
#define MyOutputDir MyRepoRoot + "\dist\installer"
#if GetEnv("FORMULA_OCR_CHINESE_ISL") != ""
#define MyChineseMessages GetEnv("FORMULA_OCR_CHINESE_ISL")
#else
#define MyChineseMessages SourcePath + "ChineseSimplified.isl"
#endif

[Setup]
AppId={{A97E48D9-8D25-46E3-9454-7100DF0AB47C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallFilesDir={app}\app
LicenseFile={#MyRepoRoot}\LICENSE
OutputDir={#MyOutputDir}
OutputBaseFilename=FormulaOCRSetup-{#MyAppVersion}
SetupIconFile={#MyRepoRoot}\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "{#MyChineseMessages}"

[CustomMessages]
english.UninstallUserData=Also remove downloaded models, settings, cache, and logs from this installation's model and app folders?
chinesesimplified.UninstallUserData=是否同时删除本安装目录 model 和 app 文件夹中的模型、设置、缓存和日志？

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyBuildDir}\*"; DestDir: "{app}"; Excludes: "app\cache\*,app\logs\*,app\settings.json,app\.storage-v2,app\.migration.lock,app\.settings.json.lock,model\*"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#MyRepoRoot}\LICENSE"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#MyRepoRoot}\NOTICE.md"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#MyRepoRoot}\README.md"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "{#MyRepoRoot}\docs\*.md"; DestDir: "{app}\app\docs"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  DeleteUserDataOnUninstall: Boolean;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ExitCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { Never delete _internal before moving any offline models it contains. }
    if not Exec(ExpandConstant('{app}\{#MyAppExeName}'), '--migrate-data --install-storage',
      ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ExitCode) then
      RaiseException('Unable to migrate FormulaOCR data. Original files were retained.');
    if ExitCode <> 0 then
      RaiseException('Model migration failed. Original files were retained; see app\logs.');
    DelTree(ExpandConstant('{app}\_internal'), True, True, True);
    DeleteFile(ExpandConstant('{app}\LICENSE'));
    DeleteFile(ExpandConstant('{app}\NOTICE.md'));
    DeleteFile(ExpandConstant('{app}\README.md'));
    if FileExists(ExpandConstant('{app}\app\unins000.exe')) then
    begin
      DeleteFile(ExpandConstant('{app}\unins000.exe'));
      DeleteFile(ExpandConstant('{app}\unins000.dat'));
      DeleteFile(ExpandConstant('{app}\unins000.msg'));
    end;
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  DeleteUserDataOnUninstall := False;
  if not UninstallSilent then
    DeleteUserDataOnUninstall := MsgBox(
      CustomMessage('UninstallUserData'),
      mbConfirmation,
      MB_YESNO or MB_DEFBUTTON2
    ) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and DeleteUserDataOnUninstall then
  begin
    DelTree(ExpandConstant('{app}\model'), True, True, True);
    DelTree(ExpandConstant('{app}\app'), True, True, True);
  end;
end;
