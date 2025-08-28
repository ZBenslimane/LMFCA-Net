#!/usr/bin/env bash


### First, process any --config options so that config file variables have lower priority
for (( argpos=1; argpos<$#; argpos++ )); do
  if [ "${!argpos}" == "--config" ]; then
    argpos_plus1=$(( argpos + 1 ))
    config=${!argpos_plus1}
    if [ ! -r "$config" ]; then
      echo "$0: Cannot read config file '$config'" >&2
      exit 1
    fi
    . "$config"  # source the config file.
  fi
done

### Now process the command-line options.
while true; do
  [ -z "${1:-}" ] && break  # exit loop if there are no arguments.
  case "$1" in
    --help|-h)
      if [ -z "${help_message:-}" ]; then
        echo "No help available." 1>&2
      else
        printf "%s\n" "$help_message" 1>&2
      fi
      exit 0 ;;
    --*=*)
      # Option of the form --option=value is not supported.
      echo "$0: options must be of the form --name value, got '$1'" 1>&2
      exit 1 ;;
    --*)
      # Remove the leading '--' and convert '-' to '_'.
      name=$(echo "$1" | sed 's/^--//' | sed 's/-/_/g')
      # Check that the variable is already defined (if not, it's an error).
      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1
      oldval=$(eval echo "\$$name")
      # Determine if the variable appears to be boolean.
      if [ "$oldval" = "true" ] || [ "$oldval" = "false" ]; then
        was_bool=true
      else
        was_bool=false
      fi
      shift
      if [ -z "${1:-}" ]; then
        echo "$0: missing argument for option --$name" 1>&2
        exit 1
      fi
      # Set the variable to the provided value.
      eval $name=\"$1\"
      if $was_bool && [[ "$1" != "true" && "$1" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\" for --$name, got '$1'" 1>&2
        exit 1
      fi
      shift ;;
    *)
      break ;;
  esac
done

# Check for empty argument for --cmd if used.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" 1>&2 && exit 1

true  # Return success.
