#!/usr/bin/env python3

import json
import sys
import os
import time
import re

from ytmusicapi import YTMusic
from typing import Optional, Union, Iterator, Dict, List
from collections import namedtuple
from dataclasses import dataclass, field
from spotify2ytmusic import ytmusic_credentials 

SongInfo = namedtuple("SongInfo", ["title", "artist", "album"])

def save_progress(playlist_id: str, progress_file: str = "spotify2ytmusic_progress.json"):
    """Save the current playlist processing progress"""
    progress_data = {
        "last_processed_playlist": playlist_id,
        "timestamp": time.time()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(progress_file: str = "spotify2ytmusic_progress.json") -> Optional[str]:
    """Load the last processed playlist ID"""
    if not os.path.exists(progress_file):
        return None
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get("last_processed_playlist")
    except (json.JSONDecodeError, KeyError):
        return None

def request_user_intervention():
    """Request user to update raw headers and oauth2.json"""
    print("\n" + "="*80)
    print("AUTHENTICATION ERROR DETECTED")
    print("="*80)
    print("The program has encountered repeated authentication failures.")
    print("This usually means the YouTube Music authentication has expired.")
    print("\nPlease perform the following steps:")
    print("1. Update your raw_headers.txt file with fresh headers from YouTube Music")
    print("2. Run 'ytmusicapi oauth' to generate a new oauth.json file")
    print("3. Ensure both files are properly updated and saved")
    print("\nFor detailed instructions on how to get fresh headers, refer to:")
    print("https://ytmusicapi.readthedocs.io/en/stable/setup/index.html")
    print("="*80)
    
    # Wait for user confirmation
    input("\nPress ENTER when you have completed the above steps and are ready to continue...")
    print("Resuming playlist processing...\n")


def get_ytmusic() -> YTMusic:
    """
    @@@
    """
    if not os.path.exists("oauth.json"):
        print("ERROR: No file 'oauth.json' exists in the current directory.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)

    try:
        return YTMusic("oauth.json")
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: JSON Decode error while trying start YTMusic: {e}")
        print("       This typically means a problem with a 'oauth.json' file.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)


def _ytmusic_create_playlist(
    yt: YTMusic, title: str, description: str, privacy_status: str = "PRIVATE"
) -> str:
    """Wrapper on ytmusic.create_playlist

    This wrapper does retries with back-off because sometimes YouTube Music will
    rate limit requests or otherwise fail.

    privacy_status can be: PRIVATE, PUBLIC, or UNLISTED
    """

    def _create(
        yt: YTMusic, title: str, description: str, privacy_status: str
    ) -> Union[str, dict]:
        exception_sleep = 5
        for _ in range(10):
            try:
                """Create a playlist on YTMusic, retrying if it fails."""
                id = yt.create_playlist(
                    title=title, description=description, privacy_status=privacy_status
                )
                return id
            except Exception as e:
                print(
                    f"ERROR: (Retrying create_playlist: {title}) {e} in {exception_sleep} seconds"
                )
                time.sleep(exception_sleep)
                exception_sleep *= 1.2

        return {
            "s2yt error": 'ERROR: Could not create playlist "{title}" after multiple retries'
        }

    id = _create(yt, title, description, privacy_status)
    #  create_playlist returns a dict if there was an error
    if isinstance(id, dict):
        print(f"ERROR: Failed to create playlist (name: {title}): {id}")
        sys.exit(1)

    time.sleep(1)  # seems to be needed to avoid missing playlist ID error

    return id


def load_playlists_json(filename: str = "playlists.json", encoding: str = "utf-8"):
    """Load the `playlists.json` Spotify playlist file"""
    return json.load(open(filename, "r", encoding=encoding))


def create_playlist(pl_name: str, privacy_status: str = "PRIVATE") -> None:
    """Create a YTMusic playlist


    Args:
        `pl_name` (str): The name of the playlist to create. It should be different to "".

        `privacy_status` (str: PRIVATE, PUBLIC, UNLISTED) The privacy setting of created playlist.
    """
    yt = get_ytmusic()

    id = _ytmusic_create_playlist(
        yt, title=pl_name, description=pl_name, privacy_status=privacy_status
    )
    print(f"Playlist ID: {id}")


def iter_spotify_liked_albums(
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
) -> Iterator[SongInfo]:
    """Songs from liked albums on Spotify."""
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    if "albums" not in spotify_pls:
        return None

    for album in [x["album"] for x in spotify_pls["albums"]]:
        for track in album["tracks"]["items"]:
            yield SongInfo(track["name"], track["artists"][0]["name"], album["name"])


def iter_spotify_playlist(
    src_pl_id: Optional[str] = None,
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
    reverse_playlist: bool = True,
) -> Iterator[SongInfo]:
    """Songs from a specific album ("Liked Songs" if None)

    Args:
        `src_pl_id` (Optional[str], optional): The ID of the source playlist. Defaults to None.
        `spotify_playlist_file` (str, optional): The path to the playlists backup files. Defaults to "playlists.json".
        `spotify_encoding` (str, optional): Characters encoding. Defaults to "utf-8".
        `reverse_playlist` (bool, optional): Is the playlist reversed when loading?  Defaults to True.

    Yields:
        Iterator[SongInfo]: The song's information
    """
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    def find_spotify_playlist(spotify_pls: Dict, src_pl_id: Union[str, None]) -> Dict:
        """Return the spotify playlist that matches the `src_pl_id`.

        Args:
            `spotify_pls`: The playlist datastrcuture saved by spotify-backup.
            `src_pl_id`: The ID of a playlist to find, or None for the "Liked Songs" playlist.
        """
        for src_pl in spotify_pls["playlists"]:
            if src_pl_id is None and str(src_pl.get("name")) == "Liked Songs":
                return src_pl
            if src_pl_id is not None and str(src_pl.get("id")) == src_pl_id:
                return src_pl
        raise ValueError(f"Could not find Spotify playlist {src_pl_id}")

    src_pl = find_spotify_playlist(spotify_pls, src_pl_id)
    src_pl_name = src_pl["name"]

    print(f"== Spotify Playlist: {src_pl_name}")

    pl_tracks = src_pl["tracks"]
    if reverse_playlist:
        pl_tracks = reversed(pl_tracks)

    for src_track in pl_tracks:
        if src_track["track"] is None:
            print(
                f"WARNING: Spotify track seems to be malformed, Skipping.  Track: {src_track!r}"
            )
            continue

        try:
            src_album_name = src_track["track"]["album"]["name"]
            src_track_artist = src_track["track"]["artists"][0]["name"]
        except TypeError as e:
            print(f"ERROR: Spotify track seems to be malformed.  Track: {src_track!r}")
            raise e
        src_track_name = src_track["track"]["name"]

        yield SongInfo(src_track_name, src_track_artist, src_album_name)


def get_playlist_id_by_name(yt: YTMusic, title: str) -> Optional[str]:
    """Look up a YTMusic playlist ID by name.

    Args:
        `yt` (YTMusic): _description_
        `title` (str): _description_

    Returns:
        Optional[str]: The playlist ID or None if not found.
    """
    #  ytmusicapi seems to run into some situations where it gives a Traceback on listing playlists
    #  https://github.com/sigma67/ytmusicapi/issues/539
    
    exception_sleep = 5
    max_retries = 5
    current_yt_instance = yt  # Use the passed-in instance initially

    for attempt in range(max_retries):
        try:
            playlists_list = current_yt_instance.get_library_playlists(limit=5000)

            if playlists_list is not None:
                # Successfully got a list (even if empty)
                for pl in playlists_list:
                    if pl["title"] == title:
                        return pl["playlistId"]
                return None # Playlist list successfully retrieved, but the specific title was not found.

            # playlists_list is None, means the API call effectively failed to return data
            print(f"Warning: (Attempt {attempt + 1}/{max_retries}) yt.get_library_playlists returned None for playlist '{title}'. Retrying in {exception_sleep}s...")
            # Proceed to retry logic

        except TypeError as e:
            # This is the specific error the user wants to catch and retry
            print(f"Warning: (Attempt {attempt + 1}/{max_retries}) Encountered TypeError for playlist '{title}': {e}. Retrying in {exception_sleep}s...")
            # Proceed to retry logic
        except KeyError as e: 
            # This is the pre-existing specific ytmusicapi bug handling for KeyError
            print("=" * 60)
            print(f"Attempting to look up playlist '{title}' failed with KeyError: {e}")
            print(
                "This is a bug in ytmusicapi that prevents 'copy_all_playlists' from working."
            )
            print(
                "You will need to manually copy playlists using s2yt_list_playlists and s2yt_copy_playlist"
            )
            print(
                "until this bug gets resolved.  Try `pip install --upgrade ytmusicapi` just to verify"
            )
            print("you have the latest version of that library.")
            print("=" * 60)
            raise # Re-raise for this specific known bug, no retry for this particular exception.

        # Common retry logic for TypeError or if playlists_list was None
        if attempt < max_retries - 1:
            time.sleep(exception_sleep)
            exception_sleep = min(exception_sleep * 1.2, 60)  # Exponential backoff with a cap
            print(f"Re-initializing YTMusic before next attempt for '{title}'...")
            try:
                # Assuming ytmusic_credentials.setup_ytmusic_with_raw_headers() prepares for re-auth
                # And get_ytmusic() returns a fresh, working instance.
                ytmusic_credentials.setup_ytmusic_with_raw_headers() 
                yt = get_ytmusic() # Get a fresh instance for the next try
            except Exception as reinit_e:
                print(f"ERROR: Failed to re-initialize YTMusic during retry for '{title}': {reinit_e}")
                # If re-initialization fails, we might want to break or return None early.
                # For now, let it try the next iteration with potentially stale 'current_yt_instance',
                # or if get_ytmusic() failed, it might raise an exception that stops the process.
        else:
            # All retries exhausted - request user intervention
            print(f"ERROR: Failed to retrieve playlist list to find '{title}' after {max_retries} retries due to TypeError or None response.")
            request_user_intervention()
            
            # After user intervention, try once more with fresh credentials
            try:
                ytmusic_credentials.setup_ytmusic_with_raw_headers() 
                yt = get_ytmusic()
                playlists_list = yt.get_library_playlists(limit=5000)
                
                if playlists_list is not None:
                    for pl in playlists_list:
                        if pl["title"] == title:
                            return pl["playlistId"]
                    return None
                else:
                    print(f"ERROR: Still unable to retrieve playlist list after user intervention for '{title}'")
                    return None
            except Exception as e:
                print(f"ERROR: Failed to retrieve playlist list even after user intervention for '{title}': {e}")
                return None

    # Fallback if loop completes (e.g. if max_retries is 0, though it's 5 here)
    return None


@dataclass
class ResearchDetails:
    query: Optional[str] = field(default=None)
    songs: Optional[List[Dict]] = field(default=None)
    suggestions: Optional[List[str]] = field(default=None)


def lookup_song(
    yt: YTMusic,
    track_name: str,
    artist_name: str,
    album_name,
    yt_search_algo: int,
    details: Optional[ResearchDetails] = None,
) -> dict:
    """Look up a song on YTMusic

    Given the Spotify track information, it does a lookup for the album by the same
    artist on YTMusic, then looks at the first 3 hits looking for a track with exactly
    the same name. In the event that it can't find that exact track, it then does
    a search of songs for the track name by the same artist and simply returns the
    first hit.

    The idea is that finding the album and artist and then looking for the exact track
    match will be more likely to be accurate than searching for the song and artist and
    relying on the YTMusic yt_search_algorithm to figure things out, especially for short tracks
    that might have many contradictory hits like "Survival by Yes".

    Args:
        `yt` (YTMusic)
        `track_name` (str): The name of the researched track
        `artist_name` (str): The name of the researched track's artist
        `album_name` (str): The name of the researched track's album
        `yt_search_algo` (int): 0 for exact matching, 1 for extended matching (search past 1st result), 2 for approximate matching (search in videos)
        `details` (ResearchDetails): If specified, more information about the search and the response will be populated for use by the caller.

    Raises:
        ValueError: If no track is found, it returns an error

    Returns:
        dict: The infos of the researched song
    """
    albums = yt.search(query=f"{album_name} by {artist_name}", filter="albums")
    for album in albums[:3]:
        # print(album)
        # print(f"ALBUM: {album['browseId']} - {album['title']} - {album['artists'][0]['name']}")

        try:
            for track in yt.get_album(album["browseId"])["tracks"]:
                if track["title"] == track_name:
                    return track
            # print(f"{track['videoId']} - {track['title']} - {track['artists'][0]['name']}")
        except Exception as e:
            print(f"Unable to lookup album ({e}), continuing...")

    query = f"{track_name} by {artist_name}"
    if details:
        details.query = query
        details.suggestions = yt.get_search_suggestions(query=query)
    songs = yt.search(query=query, filter="songs")

    match yt_search_algo:
        case 0:
            if details:
                details.songs = songs
            return songs[0]

        case 1:
            for song in songs:
                if (
                    song["title"] == track_name
                    and song["artists"][0]["name"] == artist_name
                    and song["album"]["name"] == album_name
                ):
                    return song
                # print(f"SONG: {song['videoId']} - {song['title']} - {song['artists'][0]['name']} - {song['album']['name']}")

            raise ValueError(
                f"Did not find {track_name} by {artist_name} from {album_name}"
            )

        case 2:
            #  This would need to do fuzzy matching
            for song in songs:
                # Remove everything in brackets in the song title
                song_title_without_brackets = re.sub(r"[\[(].*?[])]", "", song["title"])
                if (
                    (
                        song_title_without_brackets == track_name
                        and song["album"]["name"] == album_name
                    )
                    or (song_title_without_brackets == track_name)
                    or (song_title_without_brackets in track_name)
                    or (track_name in song_title_without_brackets)
                ) and (
                    song["artists"][0]["name"] == artist_name
                    or artist_name in song["artists"][0]["name"]
                ):
                    return song

            # Finds approximate match
            # This tries to find a song anyway. Works when the song is not released as a music but a video.
            else:
                track_name = track_name.lower()
                first_song_title = songs[0]["title"].lower()
                if (
                    track_name not in first_song_title
                    or songs[0]["artists"][0]["name"] != artist_name
                ):  # If the first song is not the one we are looking for
                    print("Not found in songs, searching videos")
                    new_songs = yt.search(
                        query=f"{track_name} by {artist_name}", filter="videos"
                    )  # Search videos

                    # From here, we search for videos reposting the song. They often contain the name of it and the artist. Like with 'Nekfeu - Ecrire'.
                    for new_song in new_songs:
                        new_song_title = new_song[
                            "title"
                        ].lower()  # People sometimes mess up the capitalization in the title
                        if (
                            track_name in new_song_title
                            and artist_name in new_song_title
                        ) or (track_name in new_song_title):
                            print("Found a video")
                            return new_song
                    else:
                        # Basically we only get here if the song isn't present anywhere on YouTube
                        raise ValueError(
                            f"Did not find {track_name} by {artist_name} from {album_name}"
                        )
                else:
                    return songs[0]


def copier(
    src_tracks: Iterator[SongInfo],
    dst_pl_id: Optional[str] = None,
    dry_run: bool = False,
    track_sleep: float = 0.1,
    yt_search_algo: int = 0,
    *,
    yt: Optional[YTMusic] = None,
):
    """
    @@@
    """
    if yt is None:
        yt = get_ytmusic()

    if dst_pl_id is not None:
        try:
            yt_pl = yt.get_playlist(playlistId=dst_pl_id)
        except Exception as e:
            print(f"ERROR: Unable to find YTMusic playlist {dst_pl_id}: {e}")
            print(
                "       Make sure the YTMusic playlist ID is correct, it should be something like "
            )
            print("      'PL_DhcdsaJ7echjfdsaJFhdsWUd73HJFca'")
            sys.exit(1)
        print(f"== Youtube Playlist: {yt_pl['title']}")

    tracks_added_set = set()
    duplicate_count = 0
    error_count = 0

    for src_track in src_tracks:
        print(f"Spotify:   {src_track.title} - {src_track.artist} - {src_track.album}")

        try:
            dst_track = lookup_song(
                yt, src_track.title, src_track.artist, src_track.album, yt_search_algo
            )
        except Exception as e:
            print(f"ERROR: Unable to look up song on YTMusic: {e}")
            error_count += 1
            continue

        yt_artist_name = "<Unknown>"
        if "artists" in dst_track and len(dst_track["artists"]) > 0:
            yt_artist_name = dst_track["artists"][0]["name"]
        print(
            f"  Youtube: {dst_track['title']} - {yt_artist_name} - {dst_track['album'] if 'album' in dst_track else '<Unknown>'}"
        )

        if dst_track["videoId"] in tracks_added_set:
            print("(DUPLICATE, this track has already been added)")
            duplicate_count += 1
        tracks_added_set.add(dst_track["videoId"])

        if not dry_run:
            exception_sleep = 5
            for _ in range(10):
                try:
                    if dst_pl_id is not None:
                        yt.add_playlist_items(
                            playlistId=dst_pl_id,
                            videoIds=[dst_track["videoId"]],
                            duplicates=False,
                        )
                    else:
                        yt.rate_song(dst_track["videoId"], "LIKE")
                    break
                except Exception as e:
                    print(
                        f"ERROR: (Retrying add_playlist_items: {dst_pl_id} {dst_track['videoId']}) {e} in {exception_sleep} seconds"
                    )
                    time.sleep(exception_sleep)
                    exception_sleep *= 1.2
                    ytmusic_credentials.setup_ytmusic_with_raw_headers() 
                    yt = get_ytmusic()  # Reinitialize YTMusic to refresh headers

        if track_sleep:
            time.sleep(track_sleep)

    print()
    print(
        f"Added {len(tracks_added_set)} tracks, encountered {duplicate_count} duplicates, {error_count} errors"
    )


def copy_playlist(
    spotify_playlist_id: str,
    ytmusic_playlist_id: str,
    spotify_playlists_encoding: str = "utf-8",
    dry_run: bool = False,
    track_sleep: float = 0.1,
    yt_search_algo: int = 0,
    reverse_playlist: bool = True,
    privacy_status: str = "PRIVATE",
):
    """
    Copy a Spotify playlist to a YTMusic playlist
    @@@
    """
    print("Using search algo n°: ", yt_search_algo)
    yt = get_ytmusic()
    pl_name: str = ""

    if ytmusic_playlist_id.startswith("+"):
        pl_name = ytmusic_playlist_id[1:]

        ytmusic_playlist_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={ytmusic_playlist_id}")

    if ytmusic_playlist_id is None:
        if pl_name == "":
            print("No playlist name or ID provided, creating playlist...")
            spotify_pls: dict = load_playlists_json()
            for pl in spotify_pls["playlists"]:
                if len(pl.keys()) > 3 and pl["id"] == spotify_playlist_id:
                    pl_name = pl["name"]

        ytmusic_playlist_id = _ytmusic_create_playlist(
            yt,
            title=pl_name,
            description=pl_name,
            privacy_status=privacy_status,
        )

        #  create_playlist returns a dict if there was an error
        if isinstance(ytmusic_playlist_id, dict):
            print(f"ERROR: Failed to create playlist: {ytmusic_playlist_id}")
            sys.exit(1)
        print(f"NOTE: Created playlist '{pl_name}' with ID: {ytmusic_playlist_id}")

    copier(
        iter_spotify_playlist(
            spotify_playlist_id,
            spotify_encoding=spotify_playlists_encoding,
            reverse_playlist=reverse_playlist,
        ),
        ytmusic_playlist_id,
        dry_run,
        track_sleep,
        yt_search_algo,
        yt=yt,
    )


def copy_all_playlists(
    track_sleep: float = 0.1,
    dry_run: bool = False,
    spotify_playlists_encoding: str = "utf-8",
    yt_search_algo: int = 0,
    reverse_playlist: bool = True,
    privacy_status: str = "PRIVATE",
):
    """
    Copy all Spotify playlists (except Liked Songs) to YTMusic playlists
    """
    spotify_pls = load_playlists_json()
    yt = get_ytmusic()

    # Check for existing progress and ask user if they want to resume
    last_processed = load_progress()
    start_from_beginning = True
    
    if last_processed:
        print(f"\nFound previous progress: last processed playlist ID was '{last_processed}'")
        while True:
            user_choice = input("Do you want to resume from where you left off? (yes/no): ").lower()
            if user_choice in ["yes", "y"]:
                start_from_beginning = False
                print(f"Resuming from playlist after '{last_processed}'...")
                break
            elif user_choice in ["no", "n"]:
                print("Starting from the beginning...")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    # Flag to track if we should start processing (used when resuming)
    should_process = start_from_beginning
    
    for src_pl in spotify_pls["playlists"]:
        if str(src_pl.get("name")) == "Liked Songs":
            continue

        pl_name = src_pl["name"]
        if pl_name == "":
            pl_name = f"Unnamed Spotify Playlist {src_pl['id']}"

        # If resuming, skip playlists until we reach the one after the last processed
        if not start_from_beginning and not should_process:
            if src_pl["id"] == last_processed:
                should_process = True  # Start processing from the next playlist
            continue
        elif not start_from_beginning and src_pl["id"] == last_processed:
            continue  # Skip the last processed playlist itself

        print(f"\n=== Processing playlist: {pl_name} (ID: {src_pl['id']}) ===")
        
        dst_pl_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={dst_pl_id}")

        spotify_track_count = len(src_pl["tracks"])

        if dst_pl_id is not None:
            yt_track_count = "unknown"
            try:
                yt_playlist_details = yt.get_playlist(playlistId=dst_pl_id, limit=1)
                yt_track_count = yt_playlist_details.get('trackCount', "unknown")
            except Exception as e:
                print(f"Warning: Could not fetch track count for existing YouTube Music playlist '{pl_name}' (ID: {dst_pl_id}): {e}")

            print(f"Playlist '{pl_name}' already exists on YouTube Music.")
            print(f"  Spotify playlist has {spotify_track_count} songs.")
            if yt_track_count != "unknown":
                print(f"  YouTube Music playlist has {yt_track_count} songs.")
            else:
                print(f"  Could not determine song count for YouTube Music playlist.")
            
            should_skip_playlist = False
            while True:
                user_choice = input("Do you want to skip this playlist? (yes/no): ").lower()
                if user_choice in ["yes", "y"]:
                    print(f"Skipping playlist '{pl_name}'.")
                    print("\nPlaylist done!\n")  # Maintain consistent output for each playlist processed
                    should_skip_playlist = True
                    break # Exit the while loop
                elif user_choice in ["no", "n"]:
                    print(f"Proceeding to update playlist '{pl_name}'.")
                    break # Exit the while loop
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            
            if should_skip_playlist:
                continue # Move to the next Spotify playlist in the outer loop
        
        # If we haven't 'continue'd (because playlist didn't exist or user chose not to skip),
        # the original logic for creation (if needed) and copying will execute.
        if dst_pl_id is None:
            dst_pl_id = _ytmusic_create_playlist(
                yt, title=pl_name, description=pl_name, privacy_status=privacy_status
            )

            #  create_playlist returns a dict if there was an error
            if isinstance(dst_pl_id, dict):
                print(f"ERROR: Failed to create playlist: {dst_pl_id}")
                sys.exit(1)
            print(f"NOTE: Created playlist '{pl_name}' with ID: {dst_pl_id}")

        copier(
            iter_spotify_playlist(
                src_pl["id"],
                spotify_encoding=spotify_playlists_encoding,
                reverse_playlist=reverse_playlist,
            ),
            dst_pl_id,
            dry_run,
            track_sleep,
            yt_search_algo,
        )
        
        # Save progress after successfully processing each playlist
        save_progress(src_pl["id"])
        print(f"Progress saved: completed playlist '{pl_name}' (ID: {src_pl['id']})")
        print("\nPlaylist done!\n")

    # Clean up progress file when all playlists are done
    if os.path.exists("spotify2ytmusic_progress.json"):
        os.remove("spotify2ytmusic_progress.json")
        print("Progress file cleaned up.")
    
    print("All done!")
