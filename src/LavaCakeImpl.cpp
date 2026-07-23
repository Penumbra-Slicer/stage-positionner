// VMA owner for stage-positionner — the only TU in this binary that includes
// LavaCake/Device.hpp without LAVACAKE_NO_VMA_IMPLEMENTATION.
// All other TUs that include LavaCake headers must define
// LAVACAKE_NO_VMA_IMPLEMENTATION before the first LavaCake include.
#include <LavaCake/Device.hpp>
