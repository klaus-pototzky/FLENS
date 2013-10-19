/*
 *   Copyright (c) 2013, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_H
#define PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_H

#include <playground/cxxblas/drivers/drivers.h>
#include <string>

namespace flens {

#ifdef HAVE_OPENCL

class OpenCLEnv{

public:
    static void
    init();

    static void
    release();

    static cl_context &
    getContext();

    static cl_command_queue &
    getQueue  ();

    static cl_command_queue *
    getQueuePtr();
    
    static cl_event *
    getEventPtr();
    
    static void
    destroyStream(int _streamID);

    static int
    getStreamID();

    static void
    setStream(int _streamID);

    static void
    syncStream(int _streamID);
    
    static void
    enableSyncCopy();
    
    static void
    disableSyncCopy();
    
    static bool
    isSyncCopyEnabled();
    
    static void
    eventRecord(int _eventID);
    
    static void
    eventSynchronize(int _eventID);
    
    static std::string
    getInfo();
    
private:
    static int                   NCalls;
    static cl_platform_id        platform;
    static cl_device_id          device ;
    static cl_context_properties props[3];
    static cl_context            ctx;
    static std::map<int, cl_command_queue>  queues;
    static int                              queueID;
    static bool                             syncCopyEnabled;
    static std::map<int, cl_event>          events;
    static cl_event                         unused_event;
};

#ifdef MAIN_FILE // Only define global variables in Mainfile

// Initialisierung der statischen Membervariablen
int                             OpenCLEnv::NCalls          = 0;
cl_platform_id                  OpenCLEnv::platform        = 0;
cl_device_id                    OpenCLEnv::device          = 0;
cl_context_properties           OpenCLEnv::props[3]        = { CL_CONTEXT_PLATFORM, 0, 0 };
cl_context                      OpenCLEnv::ctx             = 0;
std::map<int, cl_command_queue> OpenCLEnv::queues          = std::map<int, cl_command_queue>();
int                             OpenCLEnv::queueID         = 0; 
bool                            OpenCLEnv::syncCopyEnabled = true; 
std::map<int, cl_event>         OpenCLEnv::events          = std::map<int, cl_event>();
cl_event                        OpenCLEnv::unused_event    = 0;  

#endif // MAIN_FILE

#endif //HAVE_OPENCL
    

} // namespace flens

#endif // PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_H
