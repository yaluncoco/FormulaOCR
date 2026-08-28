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
english.UninstallUserData=Also remove downloaded models, settings, cache, and logs from %%LOCALAPPDATA%%\FormulaOCR?
chinesesimplified.UninstallUserData=是否同时删除 %%LOCALAPPDATA%%\FormulaOCR 中已下载的模型、设置、缓存和日志？

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyBuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#MyRepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyRepoRoot}\NOTICE.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyRepoRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
var
  DeleteUserDataOnUninstall: Boolean;

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
    DelTree(ExpandConstant('{localappdata}\FormulaOCR'), True, True, True);
end;
