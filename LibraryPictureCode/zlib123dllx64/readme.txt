Zlib 1.2.3 DLL and Library for Windows 64 bits - x64

dll_x64\zlibvc.sln : replace contrib\vstudio\vc8\zlibvc.sln of zlib 1.2.3 (minor fix)
dll_x64\zlibwapi.dll : DLL of Zlib 1.2.3 for Windows 64 bits x64 (AMD64/Intel EM64T)
dll_x64\zlibwapi.lib : The Import library of the DLL
dll_x64\demo\miniunz.exe: sample which uses the DLL
dll_x64\demo\minizip.exe: sample which uses the DLL
dll_x64\demo\testzlib.exe: sample which uses the DLL
static_x64\zlibstat.lib

These binary were build using Platform SDK windows 2003 SP1 compiler

The zlib.h must be the version from zlib123.zip / zlib-1.2.3.tar.gz

Note :
  the DLLuses the WINAPI calling convention for the exported functions, and
  includes the minizip functionality.


If you want rebuild these file, for your information :

Building instructions for the DLL versions of Zlib 1.2.3
========================================================

This directory contains projects that build zlib and minizip using
Microsoft Visual C++ 7.0/7.1, and Visual C++ 2005.

You don't need to build these projects yourself. You can download the
binaries from:
  http://www.winimage.com/zLibDll

More information can be found at this site.


Build instructions for Visual Studio 7.x (32 bits)
--------------------------------------------------
- Uncompress current zlib, including all contrib/* files
- Download the crtdll library from
    http://www.winimage.com/zLibDll/crtdll.zip
  Unzip crtdll.zip to extract crtdll.lib on contrib\vstudio\vc7.
- Open contrib\vstudio\vc7\zlibvc.sln with Microsoft Visual C++ 7.x
  (Visual Studio .Net 2002 or 2003).

Build instructions for Visual Studio 2005 (32 bits or 64 bits)
--------------------------------------------------------------
- Uncompress current zlib, including all contrib/* files
- For 32 bits only: download the crtdll library from
    http://www.winimage.com/zLibDll/crtdll.zip
  Unzip crtdll.zip to extract crtdll.lib on contrib\vstudio\vc8.
- Open contrib\vstudio\vc8\zlibvc.sln with Microsoft Visual C++ 8.0

Build instructions for Visual Studio 2005 64 bits, PSDK compiler
----------------------------------------------------------------
at the time of writing this text file, Visual Studio 2005 (and 
  Microsoft Visual C++ 8.0) is on the beta 2 stage.
Using you can get the free 64 bits compiler from Platform SDK, 
  which is NOT a beta, and compile using the Visual studio 2005 IDE
see http://www.winimage.com/misc/sdk64onvs2005/ for instruction

- Uncompress current zlib, including all contrib/* files
- start Visual Studio 2005 from a platform SDK command prompt, using
  the /useenv switch
- Open contrib\vstudio\vc8\zlibvc.sln with Microsoft Visual C++ 8.0


Important
---------
- To use zlibwapi.dll in your application, you must define the
  macro ZLIB_WINAPI when compiling your application's source files.


Additional notes
----------------
- This DLL, named zlibwapi.dll, is compatible to the old zlib.dll built
  by Gilles Vollant from the zlib 1.1.x sources, and distributed at
    http://www.winimage.com/zLibDll
  It uses the WINAPI calling convention for the exported functions, and
  includes the minizip functionality. If your application needs that
  particular build of zlib.dll, you can rename zlibwapi.dll to zlib.dll.

- The new DLL was renamed because there exist several incompatible
  versions of zlib.dll on the Internet.

- There is also an official DLL build of zlib, named zlib1.dll. This one
  is exporting the functions using the CDECL convention. See the file
  win32\DLL_FAQ.txt found in this zlib distribution.

- There used to be a ZLIB_DLL macro in zlib 1.1.x, but now this symbol
  has a slightly different effect. To avoid compatibility problems, do
  not define it here.


Gilles Vollant
info@winimage.com
