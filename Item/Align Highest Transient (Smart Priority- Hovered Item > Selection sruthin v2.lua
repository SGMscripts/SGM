-- ReaScript Name: Align Highest Transient (Smart Priority: Hovered Item > Selection) to Edit Cursor + Crossfade + Auto-Select Hovered Item
-- Author: Reaper DAW Ultimate Assistant
-- Version: 2.5
-- Description:
--   Aligns the highest transient of either the hovered item or selected item(s) to the edit cursor, ensuring smooth crossfades by mimicking Reaper's native overlap handling.
--   Applies or adjusts crossfades if overlapping, preserving existing fade settings and ensuring seamless transitions.
--   If multiple selected items are on the same track, sorts them by position, keeps the leftmost on the original track, moves subsequent ones to consecutive tracks below (creating new tracks if necessary), and aligns each.
--   Handles muted items and tracks by temporarily unmuting them during peak detection, then restoring their original muted state.
--   Smart priority logic:
--     - If hovering over unselected item → move that item and select it.
--     - If hovering over a selected item while multiple are selected → handle all selected with track logic.
--     - If no hover → handle selected item(s) with track logic if multiple.

-- ====== FUNCTION: Get Peak Sample Position ======
function get_peak_sample_position(take, item, track)
    local src = reaper.GetMediaItemTake_Source(take)
    local samplerate = reaper.GetMediaSourceSampleRate(src)
    local accessor = reaper.CreateTakeAudioAccessor(take)
    local len = reaper.GetMediaSourceLength(src)
    local blocksize = 4096
    local num_channels = 1  -- Use mono mix for peak detection

    local sample_buf = reaper.new_array(blocksize)
    local max_amp = 0
    local max_sample_pos = 0

    for i = 0, len, blocksize / samplerate do
        sample_buf.clear()
        reaper.GetAudioAccessorSamples(accessor, samplerate, num_channels, i, blocksize, sample_buf)
        local samples = sample_buf.table()

        for j = 1, #samples do
            local amp = math.abs(samples[j])
            if amp > max_amp then
                max_amp = amp
                max_sample_pos = i + (j - 1) / samplerate
            end
        end
    end

    reaper.DestroyAudioAccessor(accessor)
    return max_sample_pos
end

-- ====== FUNCTION: Crossfade if Overlapping ======
function crossfade_if_overlap(item)
    local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
    local item_start = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
    local item_end = item_start + item_len
    local track = reaper.GetMediaItem_Track(item)
    local num_items = reaper.CountTrackMediaItems(track)

    for i = 0, num_items - 1 do
        local other = reaper.GetTrackMediaItem(track, i)
        if other ~= item then
            local other_start = reaper.GetMediaItemInfo_Value(other, "D_POSITION")
            local other_end = other_start + reaper.GetMediaItemInfo_Value(other, "D_LENGTH")

            if item_start < other_end and item_end > other_start then
                local overlap_start = math.max(item_start, other_start)
                local overlap_end = math.min(item_end, other_end)
                local fade_len = overlap_end - overlap_start

                if fade_len > 0 then
                    local existing_fade_in_len = reaper.GetMediaItemInfo_Value(item, "D_FADEINLEN")
                    local existing_fade_out_len = reaper.GetMediaItemInfo_Value(item, "D_FADEOUTLEN")
                    local existing_fade_in_shape = reaper.GetMediaItemInfo_Value(item, "C_FADEINSHAPE")
                    local existing_fade_out_shape = reaper.GetMediaItemInfo_Value(item, "C_FADEOUTSHAPE")
                    local other_existing_fade_in_len = reaper.GetMediaItemInfo_Value(other, "D_FADEINLEN")
                    local other_existing_fade_out_len = reaper.GetMediaItemInfo_Value(other, "D_FADEOUTLEN")
                    local other_existing_fade_in_shape = reaper.GetMediaItemInfo_Value(other, "C_FADEINSHAPE")
                    local other_existing_fade_out_shape = reaper.GetMediaItemInfo_Value(other, "C_FADEOUTSHAPE")

                    if item_start < other_start then
                        -- Item is left, other is right: fadeout on item, fadein on other
                        if existing_fade_out_len < fade_len or existing_fade_out_len == 0 then
                            reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", fade_len)
                            reaper.SetMediaItemInfo_Value(item, "C_FADEOUTSHAPE", existing_fade_out_shape > 0 and existing_fade_out_shape or 3) -- Use equal power shape for smooth crossfade
                        end
                        if other_existing_fade_in_len < fade_len or other_existing_fade_in_len == 0 then
                            reaper.SetMediaItemInfo_Value(other, "D_FADEINLEN", fade_len)
                            reaper.SetMediaItemInfo_Value(other, "C_FADEINSHAPE", other_existing_fade_in_shape > 0 and other_existing_fade_in_shape or 3) -- Use equal power shape
                        end
                    else
                        -- Item is right, other is left: fadein on item, fadeout on other
                        if existing_fade_in_len < fade_len or existing_fade_in_len == 0 then
                            reaper.SetMediaItemInfo_Value(item, "D_FADEINLEN", fade_len)
                            reaper.SetMediaItemInfo_Value(item, "C_FADEINSHAPE", existing_fade_in_shape > 0 and existing_fade_in_shape or 3) -- Use equal power shape
                        end
                        if other_existing_fade_out_len < fade_len or other_existing_fade_out_len == 0 then
                            reaper.SetMediaItemInfo_Value(other, "D_FADEOUTLEN", fade_len)
                            reaper.SetMediaItemInfo_Value(other, "C_FADEOUTSHAPE", other_existing_fade_out_shape > 0 and other_existing_fade_out_shape or 3) -- Use equal power shape
                        end
                    end
                end
            end
        end
    end
end

-- ====== FUNCTION: Align One Item ======
function align_item_to_cursor(item, target_track)
    local take = reaper.GetActiveTake(item)
    if not take or reaper.TakeIsMIDI(take) then return end

    local track = reaper.GetMediaItem_Track(item)
    local track_was_muted = reaper.GetMediaTrackInfo_Value(track, "B_MUTE") > 0
    local item_was_muted = reaper.GetMediaItemInfo_Value(item, "B_MUTE") > 0

    if track_was_muted then
        reaper.SetMediaTrackInfo_Value(track, "B_MUTE", 0)
    end
    if item_was_muted then
        reaper.SetMediaItemInfo_Value(item, "B_MUTE", 0)
    end

    local edit_cursor = reaper.GetCursorPosition()
    local peak_time = get_peak_sample_position(take, item, track)
    local new_start = edit_cursor - peak_time

    -- Move item to align peak with edit cursor
    reaper.SetMediaItemPosition(item, new_start, true)

    if target_track then
        reaper.MoveMediaItemToTrack(item, target_track)
    end

    if track_was_muted then
        reaper.SetMediaTrackInfo_Value(track, "B_MUTE", 1)
    end
    if item_was_muted then
        reaper.SetMediaItemInfo_Value(item, "B_MUTE", 1)
    end

    crossfade_if_overlap(item)
end

-- ====== FUNCTION: Is Item Selected ======
function is_item_selected(item)
    return reaper.IsMediaItemSelected(item)
end

-- ====== FUNCTION: Handle Multiple Selected ======
function handle_multiple_selected()
    local count_sel = reaper.CountSelectedMediaItems(0)
    local items = {}
    for i = 0, count_sel - 1 do
        items[i + 1] = reaper.GetSelectedMediaItem(0, i)
    end

    if #items < 2 then
        align_item_to_cursor(items[1])
        return
    end

    local track = reaper.GetMediaItem_Track(items[1])
    local same_track = true
    for _, item in ipairs(items) do
        if reaper.GetMediaItem_Track(item) ~= track then
            same_track = false
            break
        end
    end

    if same_track then
        -- Sort items by position (leftmost first)
        table.sort(items, function(a, b)
            return reaper.GetMediaItemInfo_Value(a, "D_POSITION") < reaper.GetMediaItemInfo_Value(b, "D_POSITION")
        end)

        local current_idx = reaper.GetMediaTrackInfo_Value(track, "IP_TRACKNUMBER") - 1  -- 0-based index
        for i, item in ipairs(items) do
            if i == 1 then
                align_item_to_cursor(item)
            else
                local next_idx = current_idx + i - 1
                local target_track = reaper.GetTrack(0, next_idx)
                if not target_track then
                    reaper.InsertTrackAtIndex(next_idx, true)
                    target_track = reaper.GetTrack(0, next_idx)
                end
                align_item_to_cursor(item, target_track)
            end
        end
    else
        -- Different tracks, align each as is
        for _, item in ipairs(items) do
            align_item_to_cursor(item)
        end
    end
end

-- ====== MAIN ======
function main()
    reaper.Undo_BeginBlock()
    local count_sel = reaper.CountSelectedMediaItems(0)
    local hovered_item = nil

    -- Get hovered item (requires SWS)
    if reaper.BR_GetMouseCursorContext then
        reaper.BR_GetMouseCursorContext()
        hovered_item = reaper.BR_GetMouseCursorContext_Item()
    end

    if hovered_item then
        local hovered_is_selected = is_item_selected(hovered_item)

        if hovered_is_selected and count_sel > 1 then
            -- 🖱️ Hovering over one of multiple selected items → handle all selected with track logic
            handle_multiple_selected()
        else
            -- 🖱️ Hovering over an unselected item → move hovered item only
            reaper.Main_OnCommand(40289, 0) -- Unselect all items
            reaper.SetMediaItemSelected(hovered_item, true)
            align_item_to_cursor(hovered_item)
        end

    elseif count_sel == 1 then
        -- 🎯 One selected item, no hover
        local item = reaper.GetSelectedMediaItem(0, 0)
        align_item_to_cursor(item)

    elseif count_sel > 1 then
        -- 📦 Multiple selected items, no hover
        handle_multiple_selected()
    else
        -- 🚫 No hovered or selected item
        reaper.ShowMessageBox("No item selected or under mouse cursor.", "Info", 0)
    end

    reaper.UpdateArrange()
    reaper.Undo_EndBlock("Align highest transient (smart hover/selection logic) + crossfade + auto-select", -1)
end

main()
