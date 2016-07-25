#ifndef __CALTARGET_H__
#define __CALTARGET_H__

/** Device Kernel ISA */
typedef enum CALtargetEnum {
    CAL_TARGET_600,                 /**< R600 GPU ISA */
    CAL_TARGET_610,                 /**< RV610 GPU ISA */
    CAL_TARGET_630,                 /**< RV630 GPU ISA */
    CAL_TARGET_670,                 /**< RV670 GPU ISA */
    CAL_TARGET_7XX,                 /**< R700 class GPU ISA */
    CAL_TARGET_770,                 /**< RV770 GPU ISA */
    CAL_TARGET_710,                 /**< RV710 GPU ISA */
    CAL_TARGET_730,                 /**< RV730 GPU ISA */
    CAL_TARGET_CYPRESS,             /**< CYPRESS GPU ISA */
    CAL_TARGET_JUNIPER,             /**< JUNIPER GPU ISA */
    CAL_TARGET_REDWOOD,             /**< REDWOOD GPU ISA */
    CAL_TARGET_CEDAR,               /**< CEDAR GPU ISA */
//##BEGIN_PRIVATE##
    CAL_TARGET_SUMO,                /**< SUMO GPU ISA */
    CAL_TARGET_SUPERSUMO,           /**< SUPERSUMO GPU ISA */
    CAL_TARGET_WRESTLER,            /**< WRESTLER GPU ISA */
    CAL_TARGET_CAYMAN,              /**< CAYMAN GPU ISA */
    CAL_TARGET_KAUAI,               /**< KAUAI GPU ISA */
    CAL_TARGET_BARTS ,              /**< BARTS GPU ISA */
    CAL_TARGET_TURKS ,              /**< TURKS GPU ISA */
    CAL_TARGET_CAICOS,              /**< CAICOS GPU ISA */
    CAL_TARGET_TAHITI,              /**< TAHITI GPU ISA*/
    CAL_TARGET_PITCAIRN,            /**< PITCAIRN GPU ISA*/
    CAL_TARGET_CAPEVERDE,           /**< CAPE VERDE GPU ISA*/
    CAL_TARGET_DEVASTATOR,          /**< DEVASTATOR GPU ISA*/
    CAL_TARGET_SCRAPPER,            /**< SCRAPPER GPU ISA*/
    CAL_TARGET_OLAND,               /**< OLAND GPU ISA*/
    CAL_TARGET_BONAIRE,             /**< BONAIRE GPU ISA*/
    CAL_TARGET_SPECTRE,             /**< KAVERI1 GPU ISA*/
    CAL_TARGET_SPOOKY,              /**< KAVERI2 GPU ISA*/
    CAL_TARGET_KALINDI,             /**< KALINDI GPU ISA*/
    CAL_TARGET_HAINAN,              /**< HAINAN GPU ISA*/
    CAL_TARGET_HAWAII,              /**< HAWAII GPU ISA*/
    CAL_TARGET_ICELAND,             /**< ICELAND GPU ISA*/
    CAL_TARGET_TONGA,               /**< TONGA GPU ISA*/
    CAL_TARGET_GODAVARI,            /**< MULLINS GPU ISA*/
    CAL_TARGET_FIJI,                /**< FIJI GPU ISA*/
    CAL_TARGET_CARRIZO,             /**< CARRIZO GPU ISA*/
    CAL_TARGET_ELLESMERE,           /**< ELLESMERE GPU ISA*/
    CAL_TARGET_BAFFIN,             /**< BAFFIN GPU ISA*/
    CAL_TARGET_GREENLAND,          /**< GREENLAND GPU ISA*/
    CAL_TARGET_STONEY,             /**< STONEY GPU ISA*/
    CAL_TARGET_LEXA,               /**< LEXA GPU ISA*/
    CAL_TARGET_LAST = CAL_TARGET_LEXA, /**< last */
//##END_PRIVATE##
} CALtarget;

#endif
