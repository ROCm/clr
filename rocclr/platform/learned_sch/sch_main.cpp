#include <sys/time.h>
#include "sch_main.hpp"
#include "env.hpp"
#include "agent.hpp"
#include <pthread.h>
static pthread_once_t LschInitCallOnce = PTHREAD_ONCE_INIT;

void *lsch_main(void *args)
{
    Observe_st obs;
    obs.current_queue = -1;
    AMDresult res = ObserveState(&obs);
    struct timeval start, end;
    int count = 0;
    if (res) {
    }; // TODO

    Agent agent(obs);

    while (1) {
        res = ObserveState(&obs);

        if (obs.kernelNum == 0) {
            continue;
        }

        agent.schedule(obs);
    }
}

void createLschThread()
{
    pthread_t thread;
    int result = pthread_create(&thread, NULL, lsch_main, NULL);
    if (result != 0) {
        std::cout << "pthread_create failed: " << result << std::endl;
        exit(0);
    }
    pthread_detach(thread);
}

void lsch_init()
{
    pthread_once(&LschInitCallOnce, createLschThread);
}

void lsch_exit(void)
{
}
