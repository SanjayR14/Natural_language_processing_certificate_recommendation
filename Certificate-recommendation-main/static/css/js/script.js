document.getElementById('fetchButton').addEventListener('click', function() {
    fetch('/get_ip_info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('ip').textContent = data.ip;
            document.getElementById('city').textContent = data.location.city;
            document.getElementById('region').textContent = data.location.region;
            document.getElementById('country').textContent = data.location.country;
            document.getElementById('location').textContent = data.location.location;
            document.getElementById('isp').textContent = data.location.isp;

            // Update audio source
            const audio = document.getElementById('audio');
            audio.src = 'data:audio/wav;base64,' + data.audio;
        })
        .catch(error => console.error('Error:', error));
});
