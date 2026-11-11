// SPDX-License-Identifier: GPL-2.0-or-later
// Info overlay text (10x20 font, character cell coordinates), clipped to the frame.
#pragma once

#include "deflicker.hpp"

namespace deflicker {

void drawString(Frame& dst, const Format& fmt, int col, int row, const char* s);

}  // namespace deflicker
