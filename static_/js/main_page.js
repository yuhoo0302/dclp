
// 如果.select_image，那么就把其border颜色变成#10239e
select_images = document.querySelectorAll('.select_image');
submit_label = document.getElementById('submit_label');
submit_button = document.getElementById('submit');
for (var i = 0; i < select_images.length; i++) {
    select_images[i].style.borderColor = '#f0f0f0';
    select_images[i].addEventListener('click', function () {
        if (this.style.borderColor === 'rgb(240, 240, 240)') {
            for (var i = 0; i < select_images.length; i++)
                select_images[i].style.borderColor = '#f0f0f0';
            this.style.borderColor = '#304ffe';
            submit_label.style.cursor = 'pointer';
            submit_button.disabled = false;
            // 获取this的id，将id通过“_”分割，取最后一个个元素，即为图片的id
            let id = this.id.split('_').pop();
            submit_button.onclick = function () {
                window.location.href = '/main_page?image_id=' + id;
                document.getElementById('loading_div').style.display = 'flex';
            }
            let case_id = case_num[id];
            let image_id = '';
            for (let i = 0; i < case_id; i++) {
                image_id += '<div class="result_image">' +
                    '    <button class="result_image_button">' +
                    '        <img alt="" src="../static/temp/' + id + '/' + i + '.png" ' + case_id[i] + '">' +
                    '    </button>' +
                    '</div>';
            }
            document.getElementById('result_image_div').innerHTML = image_id;
        } else {
            this.style.borderColor = '#f0f0f0';
            submit_label.style.cursor = 'not-allowed';
            submit_button.disabled = true;
        }
    });
}

// id为file的input标签，当选择文件后，直接提交表单
document.getElementById('file').addEventListener('change', async function () {
    document.getElementsByTagName('form')[0].submit();
}, true);

if (diagnosis === 'Normal') {
    document.getElementById('result_normal').style.borderColor = '#339af0';
    document.getElementById('result_normal').style.background = '#339af0';
    document.getElementById('result_normal_text').style.color = '#fefefe';
} else if (diagnosis === 'CL') {
    document.getElementById('result_CL').style.borderColor = '#ff6b6b';
    document.getElementById('result_CL').style.background = '#ff6b6b';
    document.getElementById('result_CL_text').style.color = '#fefefe';
} else if (diagnosis === 'CLP') {
    document.getElementById('result_CLP').style.borderColor = '#ff6b6b';
    document.getElementById('result_CLP').style.background = '#ff6b6b';
    document.getElementById('result_CLP_text').style.color = '#fefefe';
} else if (diagnosis === 'Uncertain') {
    document.getElementById('result_uncertain').style.borderColor = '#8a8886';
    document.getElementById('result_uncertain').style.background = '#8a8886';
    document.getElementById('result_uncertain_text').style.color = '#fefefe';
}
